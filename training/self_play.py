# training/self_play.py - 并行自我对弈版
"""
优化:
1. 多进程并行对弈 (每个 worker 独立的 board + MCTS)
2. 网络用 eval() + torch.no_grad() 确保推理模式
3. 棋盘 deepcopy 改为轻量复制（board 自身实现 copy 会更快）
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple
import copy


def _worker_play_games(
    rank: int,
    num_games: int,
    network_state_dict: dict,
    device: torch.device,
    move_encoder,
    config: dict,
    board_cls,  # ChineseChessBoard 类（不能序列化实例，传类）
    result_queue: mp.Queue,
):
    """子进程工作函数: 独立初始化网络副本和 MCTS，串行下完 num_games 盘棋"""
    from model.network import ChessNet
    from model.mcts import MCTS

    # 每个进程自己的网络副本（CPU 推理也可以，但 GPU 更快）
    action_size = move_encoder.action_size
    network = ChessNet(
        num_res_blocks=config["num_res_blocks"],
        channels=config["channels"],
        action_size=action_size,
    ).to(device)

    # 处理 torch.compile 产生的 _orig_mod. 前缀
    fixed_state_dict = {}
    for k, v in network_state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        if new_k not in fixed_state_dict:
            fixed_state_dict[new_k] = v

    network.load_state_dict(fixed_state_dict, strict=False)
    network.eval()

    mcts = MCTS(network, move_encoder, config)
    board = board_cls()

    all_game_data = []
    for _ in range(num_games):
        game_data = _play_single_game(board, mcts, move_encoder)
        all_game_data.extend(game_data)

    # 把结果放入队列（自动序列化）
    result_queue.put(all_game_data)


def _play_single_game(board, mcts, move_encoder) -> List[Tuple]:
    """下一盘棋，返回 [(state, mcts_probs, reward), ...]"""
    training_data = []
    board.reset()

    while True:
        mcts_probs = mcts.search(board)

        state = board.get_state()
        training_data.append(
            {
                "state": state.copy(),
                "mcts_probs": mcts_probs.copy(),
                "player": board.current_player,
            }
        )

        # 按 MCTS 概率采样动作
        action_idx = np.random.choice(len(mcts_probs), p=mcts_probs)
        action = move_encoder.decode(action_idx)

        if action is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            action = legal_moves[np.random.randint(len(legal_moves))]

        game_over, _ = board.make_move(action)
        if game_over or len(training_data) > 200:
            break

    _, winner = board.check_game_over()

    processed = []
    for data in training_data:
        if winner == 0:
            reward = 0.0
        elif data["player"] == winner:
            reward = 1.0
        else:
            reward = -1.0
        processed.append((data["state"], data["mcts_probs"], reward))

    return processed


class SelfPlayWorker:
    """
    自我对弈管理器
    - 单 GPU 时: 用 num_workers 个子进程轮流占用 GPU（MCTS 推理是主要瓶颈，
      多进程可以掩盖 CPU 棋盘逻辑的耗时）
    - 也支持纯串行 fallback (num_workers=1)
    """

    def __init__(self, network, mcts, move_encoder, config=None):
        self.network = network
        self.mcts = mcts
        self.move_encoder = move_encoder
        self.config = config or {}
        self.num_workers = self.config.get("num_self_play_workers", 1)

    def play_games(self, board, num_games: int) -> List:
        """下完 num_games 盘棋，返回所有训练数据"""
        if self.num_workers <= 1 or not torch.cuda.is_available():
            # 串行 fallback
            return self._play_serial(board, num_games)
        else:
            return self._play_parallel(board, num_games)

    # ----------------------------------------------------------- 串行路径
    def _play_serial(self, board, num_games: int) -> List:
        self.network.eval()
        all_data = []
        for _ in range(num_games):
            game_data = _play_single_game(board, self.mcts, self.move_encoder)
            all_data.extend(game_data)
        return all_data

    # ---------------------------------------------------------- 并行路径
    def _play_parallel(self, board, num_games: int) -> List:
        """
        将 num_games 平均分配到 num_workers 个子进程。
        注意: 单 GPU 场景下多进程共享同一显卡，cuda 推理会自动排队，
        实际加速来源于 CPU 棋盘逻辑与 GPU 推理的重叠。
        """
        device = next(self.network.parameters()).device
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # 每个 worker 分配的对局数
        games_per_worker = num_games // self.num_workers
        remainder = num_games % self.num_workers

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []

        from game.board import ChineseChessBoard

        for rank in range(self.num_workers):
            n = games_per_worker + (1 if rank < remainder else 0)
            if n == 0:
                continue
            p = ctx.Process(
                target=_worker_play_games,
                args=(
                    rank,
                    n,
                    state_dict,
                    device,
                    self.move_encoder,
                    self.config,
                    ChineseChessBoard,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # 收集结果
        all_data = []
        for _ in processes:
            all_data.extend(result_queue.get())

        for p in processes:
            p.join()

        return all_data

    # 保留旧接口兼容
    def play_game(self, board) -> List[Tuple]:
        self.network.eval()
        return _play_single_game(board, self.mcts, self.move_encoder)
