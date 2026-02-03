# train.py - 完整训练入口 (参考 ChineseChess-AlphaZero)
"""
改进:
- 阶梯式学习率调度
- 详细的自我对弈日志
- 验证集监控
- TensorBoard 日志
- SGD 优化器
"""

import torch
import torch.optim as optim
import time
import os
import argparse
import sys
import json
from datetime import datetime
from game.board import ChineseChessBoard
from model.network import ChessNet, MoveEncoder, compile_network
from model.mcts import MCTS
from training.self_play import SelfPlayWorker
from training.trainer import Trainer
from config import config as default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Chinese Chess AI Training")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--config", type=str, default="", help="Path to custom config file"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override iterations"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="GPU device ID",
    )
    return parser.parse_args()


def load_config(config_path):
    config = {}
    if config_path and os.path.exists(config_path):
        exec(open(config_path).read(), {"config": config})
        print(f"Loaded config from {config_path}")
    return config


def get_learning_rate(total_steps, lr_schedule):
    """根据总步数获取当前学习率"""
    lr = None
    for step, rate in lr_schedule:
        if total_steps >= step:
            lr = rate
    return lr


def main():
    start_time = time.time()

    args = parse_args()

    device_id = args.gpu.strip()
    if device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"Using GPU: {device_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = default_config.copy()
    if args.config:
        custom_config = load_config(args.config)
        config.update(custom_config)
    if args.iterations:
        config["num_iterations"] = args.iterations

    print(f"\n{'=' * 60}")
    print(f"训练配置:")
    print(f"  - ResBlocks: {config['num_res_blocks']}, Channels: {config['channels']}")
    print(f"  - MCTS simulations: {config['num_simulations']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate schedule: {config['lr_schedule']}")
    print(f"  - Momentum: {config['momentum']}")
    print(f"  - L2 reg: {config['l2_reg']}")
    print(f"  - Validation split: {config['validation_split']}")
    print(f"{'=' * 60}\n")

    move_encoder = MoveEncoder()
    action_size = move_encoder.action_size
    print(f"Action size: {action_size}")

    network = ChessNet(
        num_res_blocks=config["num_res_blocks"],
        channels=config["channels"],
        action_size=action_size,
    ).to(device)

    network = compile_network(network)

    mcts = MCTS(network, move_encoder, config)
    self_play = SelfPlayWorker(network, mcts, move_encoder, config)
    trainer = Trainer(network, config, action_size=action_size)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    start_iteration = 0
    total_train_steps = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        sd = checkpoint.get("model_state_dict", checkpoint)
        if hasattr(network, "_orig_mod"):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        network.load_state_dict(sd, strict=False)

        if "optimizer_state_dict" in checkpoint and hasattr(
            trainer.optimizer, "load_state_dict"
        ):
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                print("Could not load optimizer state")

        start_iteration = checkpoint.get("iteration", 0) + 1
        total_train_steps = checkpoint.get("total_train_steps", 0)
        saved_config = checkpoint.get("config", {})
        if saved_config:
            print(
                f"Resuming from iteration {start_iteration}, total steps {total_train_steps}"
            )

    print(f"\nStarting training, target: {config['num_iterations']} iterations")
    print(f"Starting from iteration: {start_iteration}")
    print("=" * 60)

    log_file = open("training_log.txt", "a", encoding="utf-8")

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    for iteration in range(start_iteration, config["num_iterations"]):
        iter_start = time.time()
        current_iter = iteration + 1

        log(f"\n{'=' * 60}")
        log(f"Iteration {current_iter}/{config['num_iterations']}")

        network.eval()
        game_data, game_stats = self_play.play_games(
            ChineseChessBoard(), config["games_per_iteration"]
        )
        log(f"  对弈完成: {self_play.format_game_summary(game_stats)}")

        trainer.add_game_data(game_data)
        log(f"  Buffer size: {len(trainer.buffer)}")

        current_lr = get_learning_rate(total_train_steps, config["lr_schedule"])
        if current_lr is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = current_lr
            log(f"  Learning rate: {current_lr:.6f}")

        network.train()
        metrics = trainer.train_epoch(config["train_steps_per_iteration"])
        total_train_steps += config["train_steps_per_iteration"]

        log(
            f"  Loss: {metrics['loss']:.4f} "
            f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})"
        )
        if "val_loss" in metrics:
            log(f"  Val Loss: {metrics['val_loss']:.4f}")
        if "policy_acc" in metrics:
            log(f"  Policy Acc: {metrics['policy_acc']:.2%}")

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            log(f"  GPU Mem: {mem:.2f} GB")

        elapsed = time.time() - iter_start
        total_elapsed = time.time() - start_time
        iterations_done = iteration - start_iteration + 1
        eta = (total_elapsed / iterations_done) * (
            config["num_iterations"] - current_iter
        )

        log(
            f"  Time: {elapsed:.1f}s | Total: {total_elapsed / 3600:.2f}h | ETA: {eta / 3600:.2f}h"
        )
        log(f"  Total train steps: {total_train_steps}")

        if current_iter % config["save_model_steps"] == 0:
            path = f"checkpoints/model_iter_{current_iter}.pth"
            sd = (
                network._orig_mod.state_dict()
                if hasattr(network, "_orig_mod")
                else network.state_dict()
            )
            torch.save(
                {
                    "iteration": iteration,
                    "total_train_steps": total_train_steps,
                    "model_state_dict": sd,
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": config,
                },
                path,
            )
            log(f"  Checkpoint: {path}")

    total_time = time.time() - start_time
    log(f"\n训练完成! 总用时: {total_time / 3600:.2f} 小时")
    log(f"总训练步数: {total_train_steps}")
    log_file.close()


if __name__ == "__main__":
    main()
