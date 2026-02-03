# train.py - 优化版
"""
完整训练入口。变更:
- network 用 torch.compile 编译
- 对弈时 network.eval(), 训练时 network.train()
- Trainer 用新的预分配 buffer 接口
- 清理日志输出
"""

import torch
import time
import os

from game.board import ChineseChessBoard
from model.network import ChessNet, MoveEncoder, compile_network
from model.mcts import MCTS
from training.self_play import SelfPlayWorker
from training.trainer import Trainer
from config import config


def main():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"✓ cuDNN benchmark enabled")
        print(f"✓ TF32 enabled (if supported)")

    print(f"Config: {config}")

    # --- 初始化 ---
    move_encoder = MoveEncoder()
    action_size = move_encoder.action_size
    print(f"Action size: {action_size}")

    network = ChessNet(
        num_res_blocks=config["num_res_blocks"],
        channels=config["channels"],
        action_size=action_size,
    ).to(device)

    # torch.compile 加速 (PyTorch 2.0+)
    network = compile_network(network)

    mcts = MCTS(network, move_encoder, config)
    self_play = SelfPlayWorker(network, mcts, move_encoder, config)
    trainer = Trainer(network, config, action_size=action_size)

    board = ChineseChessBoard()
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n开始训练，目标: {config['num_iterations']} 轮")
    print("=" * 60)

    for iteration in range(config["num_iterations"]):
        iter_start = time.time()
        print(f"\n===== Iteration {iteration + 1}/{config['num_iterations']} =====")

        # ---- 自我对弈 (eval 模式: BatchNorm 用 running stats, 不更新) ----
        network.eval()
        game_data = self_play.play_games(board, config["games_per_iteration"])
        print(f"  对弈收集: {len(game_data)} 个样本")

        trainer.add_game_data(game_data)
        print(f"  Buffer size: {len(trainer.buffer)}")

        # ---- 训练 (train 模式: BatchNorm 用 batch stats) ----
        network.train()
        metrics = trainer.train_epoch(config["train_steps_per_iteration"])
        print(
            f"  Loss: {metrics['loss']:.4f} "
            f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})"
        )

        # ---- 监控 ----
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU Peak Mem: {mem:.2f} GB")

        elapsed = time.time() - iter_start
        total_elapsed = time.time() - start_time
        eta = (total_elapsed / (iteration + 1)) * (
            config["num_iterations"] - iteration - 1
        )
        print(
            f"  本轮: {elapsed:.1f}s | 总计: {total_elapsed / 3600:.2f}h | ETA: {eta / 3600:.2f}h"
        )

        # ---- Checkpoint ----
        if (iteration + 1) % 10 == 0:
            path = f"checkpoints/model_iter_{iteration + 1}.pth"
            # compiled model 需要存 _orig_mod 的 state_dict
            sd = (
                network._orig_mod.state_dict()
                if hasattr(network, "_orig_mod")
                else network.state_dict()
            )
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": sd,
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": config,
                },
                path,
            )
            print(f"  Checkpoint: {path}")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time / 3600:.2f} 小时")
    print(f"总对局数: {config['num_iterations'] * config['games_per_iteration']}")


if __name__ == "__main__":
    main()
