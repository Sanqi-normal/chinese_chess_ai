# training/self_play.py - 详细日志版 + Material Reward
"""
改进:
- 每局对弈详细日志
- 胜负统计
- 步数统计
- 投子信息
- Material reward (子力价值作为中间奖励)
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Dict, Optional
import copy
import time
from datetime import datetime


def format_move(move):
    """美化走法显示"""
    fr, fc, tr, tc = move
    col_names = "abcdefghi"
    return f"{col_names[fc]}{9 - fr}{col_names[tc]}{9 - tr}"


def calculate_material_score(board, piece_values):
    """计算当前局面子力分 (红方视角)"""
    score = 0
    for row in range(10):
        for col in range(9):
            piece = board.board[row, col]
            if piece != 0:
                score += piece_values.get(piece, 0)
    return score


class GameRecord:
    """单局对弈记录"""

    def __init__(self, game_id: int):
        self.game_id = game_id
        self.moves = []
        self.winner = None
        self.turn_count = 0
        self.duration = 0.0
        self.red_captured = []
        self.black_captured = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.resigned = False
        self.reason = ""
        self.final_material_diff = 0.0

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "moves": self.moves,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "duration_sec": self.duration,
            "red_captured": self.red_captured,
            "black_captured": self.black_captured,
            "resigned": self.resigned,
            "reason": self.reason,
            "final_material_diff": self.final_material_diff,
        }


def _worker_play_games(
    rank: int,
    num_games: int,
    network_state_dict: dict,
    device: torch.device,
    move_encoder,
    config: dict,
    board_cls,
    result_queue: mp.Queue,
    verbose: bool = False,
):
    from model.network import ChessNet
    from model.mcts import MCTS

    action_size = move_encoder.action_size
    network = ChessNet(
        num_res_blocks=config["num_res_blocks"],
        channels=config["channels"],
        action_size=action_size,
    ).to(device)

    fixed_state_dict = {}
    for k, v in network_state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        if new_k not in fixed_state_dict:
            fixed_state_dict[new_k] = v

    network.load_state_dict(fixed_state_dict, strict=False)
    network.eval()

    mcts = MCTS(network, move_encoder, config)

    all_game_data = []
    game_records = []

    base_game_id = int(time.time() * 1000) % 1000000

    for game_idx in range(num_games):
        game_id = base_game_id + game_idx
        game_record = GameRecord(game_id)
        game_data = _play_single_game(
            board_cls(), mcts, move_encoder, game_record, config, verbose
        )
        all_game_data.extend(game_data)
        game_records.append(game_record)

    result_queue.put((all_game_data, game_records))


def _play_single_game(
    board,
    mcts,
    move_encoder,
    game_record: GameRecord,
    config: dict,
    verbose: bool = False,
) -> List[Tuple]:
    """下一盘棋，返回训练数据和更新 game_record"""
    training_data = []
    board.reset()

    piece_values = config.get("piece_values", {})
    use_material_reward = config.get("use_material_reward", True)
    material_scale = config.get("material_scale", 0.01)
    win_reward = config.get("win_reward", 1.0)
    loss_reward = config.get("loss_reward", -1.0)
    draw_reward = config.get("draw_reward", 0.0)

    game_record.start_time = time.time()

    initial_material = calculate_material_score(board, piece_values)
    prev_material = initial_material

    move_number = 0
    resign_threshold = config.get("resign_threshold", -0.98)
    min_resign_turn = config.get("min_resign_turn", 80)
    enable_resign_rate = config.get("enable_resign_rate", 0.5)
    max_game_length = config.get("max_game_length", 256)

    can_resign = np.random.random() < enable_resign_rate

    while True:
        move_number += 1
        current_player = "红方" if board.current_player == 1 else "黑方"

        mcts_probs = mcts.search(board)

        if verbose and move_number <= 5:
            top_probs = sorted(enumerate(mcts_probs), key=lambda x: x[1], reverse=True)[
                :3
            ]
            top_moves = [
                (format_move(move_encoder.decode(idx)), prob)
                for idx, prob in top_probs
                if move_encoder.decode(idx) is not None
            ]

        state = board.get_state()
        game_record.moves.append(
            {
                "turn": move_number,
                "player": current_player,
            }
        )

        training_data.append(
            {
                "state": state.copy(),
                "mcts_probs": mcts_probs.copy(),
                "player": board.current_player,
            }
        )

        action_idx = np.random.choice(len(mcts_probs), p=mcts_probs)
        action = move_encoder.decode(action_idx)

        if action is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                game_record.reason = "无合法走法"
                break
            action = legal_moves[np.random.randint(len(legal_moves))]
            if verbose:
                print(f"    [警告] MCTS推荐为空，使用随机走法")

        game_record.moves[-1]["action"] = format_move(action)

        captured = board.board[action[2], action[3]]
        if captured != 0:
            piece_name = board.PIECE_NAMES.get(captured, "?")
            if board.current_player == 1:
                game_record.red_captured.append(piece_name)
            else:
                game_record.black_captured.append(piece_name)
            if verbose:
                print(f"    吃子: {piece_name}")

        game_over, _ = board.make_move(action)

        if verbose and move_number <= 10:
            print(f"  步{move_number}: {current_player} {format_move(action)}")

        if game_over or move_number >= max_game_length:
            if move_number >= max_game_length:
                game_record.reason = "达到最大步数"
            game_record.turn_count = move_number
            break

        value = mcts_probs[action_idx]
        if can_resign and move_number >= min_resign_turn and value < resign_threshold:
            if verbose:
                print(f"  [投子] 步{move_number}, 置信度={value:.3f}")
            game_record.resigned = True
            game_record.reason = f"投子认输 (conf={value:.3f})"
            game_record.turn_count = move_number
            break

    game_record.end_time = time.time()
    game_record.duration = game_record.end_time - game_record.start_time

    game_over, winner = board.check_game_over()
    game_record.winner = winner

    final_material = calculate_material_score(board, piece_values)
    game_record.final_material_diff = final_material - initial_material

    use_material = use_material_reward and material_scale > 0

    processed = []
    for data in training_data:
        player = data["player"]

        if winner == 0:
            final_reward = draw_reward
        elif winner == player:
            final_reward = win_reward
        else:
            final_reward = loss_reward

        if use_material:
            current_material = calculate_material_score(board, piece_values)
            material_diff = current_material - prev_material
            material_reward = material_diff * material_scale

            if player == 1:
                reward = material_reward + final_reward * 0.1
            else:
                reward = -material_reward + final_reward * 0.1
        else:
            reward = final_reward

        processed.append((data["state"], data["mcts_probs"], reward))

    return processed


class SelfPlayWorker:
    def __init__(self, network, mcts, move_encoder, config=None):
        self.network = network
        self.mcts = mcts
        self.move_encoder = move_encoder
        self.config = config or {}
        self.num_workers = self.config.get("num_self_play_workers", 1)

    def play_games(
        self, board, num_games: int, verbose: bool = False
    ) -> Tuple[List, Dict]:
        """下完 num_games 盘棋，返回所有训练数据和统计"""
        if self.num_workers <= 1 or not torch.cuda.is_available():
            return self._play_serial(board, num_games, verbose)
        else:
            return self._play_parallel(board, num_games, verbose)

    def _play_serial(
        self, board, num_games: int, verbose: bool = False
    ) -> Tuple[List, Dict]:
        self.network.eval()
        all_data = []
        game_records = []

        base_game_id = int(time.time() * 1000) % 1000000

        for game_idx in range(num_games):
            game_record = GameRecord(base_game_id + game_idx)
            game_data = _play_single_game(
                board, self.mcts, self.move_encoder, game_record, self.config, verbose
            )
            all_data.extend(game_data)
            game_records.append(game_record)

        stats = self._aggregate_stats(game_records)
        return all_data, stats

    def _play_parallel(
        self, board, num_games: int, verbose: bool = False
    ) -> Tuple[List, Dict]:
        device = next(self.network.parameters()).device
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

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
                    verbose,
                ),
            )
            p.start()
            processes.append(p)

        all_data = []
        all_records = []

        for _ in processes:
            data, records = result_queue.get()
            all_data.extend(data)
            all_records.extend(records)

        for p in processes:
            p.join()

        stats = self._aggregate_stats(all_records)
        return all_data, stats

    def _aggregate_stats(self, game_records: List[GameRecord]) -> Dict:
        """聚合对局统计"""
        total = len(game_records)
        if total == 0:
            return {}

        red_wins = sum(1 for r in game_records if r.winner == 1)
        black_wins = sum(1 for r in game_records if r.winner == -1)
        draws = sum(1 for r in game_records if r.winner == 0)
        resigns = sum(1 for r in game_records if r.resigned)
        avg_turns = sum(r.turn_count for r in game_records) / total
        avg_duration = sum(r.duration for r in game_records) / total
        avg_material_diff = sum(r.final_material_diff for r in game_records) / total

        all_red_captured = []
        all_black_captured = []
        for r in game_records:
            all_red_captured.extend(r.red_captured)
            all_black_captured.extend(r.black_captured)

        return {
            "total_games": total,
            "red_wins": red_wins,
            "black_wins": black_wins,
            "draws": draws,
            "red_win_rate": red_wins / total if total > 0 else 0,
            "black_win_rate": black_wins / total if total > 0 else 0,
            "resigns": resigns,
            "avg_turns": avg_turns,
            "avg_duration_sec": avg_duration,
            "total_samples": sum(r.turn_count for r in game_records),
            "captured_by_red": all_red_captured,
            "captured_by_black": all_black_captured,
            "avg_material_diff": avg_material_diff,
        }

    def format_game_summary(self, stats: Dict) -> str:
        """格式化对局统计输出"""
        if not stats:
            return "No games played"

        lines = [
            f"总对局: {stats['total_games']}",
            f"红方胜: {stats['red_wins']} ({stats['red_win_rate']:.1%})",
            f"黑方胜: {stats['black_wins']} ({stats['black_win_rate']:.1%})",
            f"平局: {stats['draws']}",
            f"投子数: {stats['resigns']}",
            f"平均步数: {stats['avg_turns']:.1f}",
            f"平均用时: {stats['avg_duration_sec']:.1f}s",
            f"子力差均值: {stats.get('avg_material_diff', 0):.1f}",
            f"总样本数: {stats['total_samples']}",
        ]

        if stats.get("captured_by_red"):
            red_counts = {}
            for p in stats["captured_by_red"]:
                red_counts[p] = red_counts.get(p, 0) + 1
            lines.append(f"红方吃子: {red_counts}")

        if stats.get("captured_by_black"):
            black_counts = {}
            for p in stats["captured_by_black"]:
                black_counts[p] = black_counts.get(p, 0) + 1
            lines.append(f"黑方吃子: {black_counts}")

        return " | ".join(lines)

    def play_game(self, board, verbose: bool = False) -> List[Tuple]:
        self.network.eval()
        game_record = GameRecord(0)
        return _play_single_game(
            board, self.mcts, self.move_encoder, game_record, self.config, verbose
        )
