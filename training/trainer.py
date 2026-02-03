# training/trainer.py - 改进版
"""
改进:
- SGD 优化器 + momentum
- 验证集支持
- TensorBoard 日志
- 训练精度监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import os
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """预分配 numpy buffer"""

    def __init__(self, capacity: int, action_size: int, state_shape: tuple):
        self.capacity = capacity
        self.action_size = action_size
        self.state_shape = state_shape

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.probs = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

        self.size = 0
        self.ptr = 0
        self.lock = threading.Lock()

    def push(self, data: list):
        with self.lock:
            for state, prob, value in data:
                self.states[self.ptr] = state
                self.probs[self.ptr] = prob
                self.values[self.ptr, 0] = value
                self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + len(data), self.capacity)

    def sample(self, batch_size: int, validation: bool = False):
        with self.lock:
            if validation:
                val_size = int(self.size * 0.02)
                indices = np.random.choice(self.size, size=val_size, replace=False)
            else:
                indices = np.random.randint(0, self.size, size=batch_size)

            s = self.states[indices].copy()
            p = self.probs[indices].copy()
            v = self.values[indices].copy()

        return (
            torch.from_numpy(s).pin_memory(),
            torch.from_numpy(p).pin_memory(),
            torch.from_numpy(v).pin_memory(),
        )

    def __len__(self):
        return self.size


class Trainer:
    def __init__(
        self, network, config, action_size: int, state_shape: tuple = (14, 10, 9)
    ):
        self.network = network
        self.device = next(network.parameters()).device

        lr_schedule = config.get("lr_schedule", [(0, 0.01)])
        initial_lr = lr_schedule[0][1] if lr_schedule else 0.01

        self.optimizer = optim.SGD(
            network.parameters(),
            lr=initial_lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.buffer = ReplayBuffer(
            capacity=config.get("buffer_size", 500000),
            action_size=action_size,
            state_shape=state_shape,
        )
        self.batch_size = config.get("batch_size", 512)
        self.val_split = config.get("validation_split", 0.02)

        log_dir = config.get("log_dir", "./logs")
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        self.use_amp = False
        self.scaler = None
        print(f"Trainer initialized: batch_size={self.batch_size}, lr={initial_lr}")

    def add_game_data(self, data: list):
        self.buffer.push(data)

    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        states, target_probs, target_values = self.buffer.sample(self.batch_size)

        states = states.to(self.device, non_blocking=True)
        target_probs = target_probs.to(self.device, non_blocking=True)
        target_values = target_values.to(self.device, non_blocking=True)

        pred_log_probs, pred_values = self.network(states)

        policy_loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))
        value_loss = nn.functional.mse_loss(pred_values, target_values)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_epoch(self, num_steps: int) -> dict:
        """训练一个 epoch，返回平均 metrics"""
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        actual = 0

        for step_idx in range(num_steps):
            m = self.train_step()
            if m["loss"] > 0:
                for k in totals:
                    totals[k] += m[k]
                actual += 1

        if actual > 0:
            totals = {k: v / actual for k, v in totals.items()}

        if self.writer:
            self.writer.add_scalar("Loss/train", totals["loss"], self.global_step)
            self.writer.add_scalar(
                "Policy_Loss/train", totals["policy_loss"], self.global_step
            )
            self.writer.add_scalar(
                "Value_Loss/train", totals["value_loss"], self.global_step
            )
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning_Rate", current_lr, self.global_step)

        if actual > 0:
            totals = {k: v / actual for k, v in totals.items()}

        return totals

    def validate(self) -> dict:
        """验证集评估"""
        if len(self.buffer) < self.batch_size:
            return {}

        states, target_probs, target_values = self.buffer.sample(
            self.batch_size, validation=True
        )

        states = states.to(self.device, non_blocking=True)
        target_probs = target_probs.to(self.device, non_blocking=True)
        target_values = target_values.to(self.device, non_blocking=True)

        self.network.eval()
        with torch.no_grad():
            pred_log_probs, pred_values = self.network(states)

        policy_loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))
        value_loss = nn.functional.mse_loss(pred_values, target_values)
        loss = policy_loss + value_loss

        pred_actions = torch.argmax(pred_log_probs, dim=1)
        target_actions = torch.argmax(target_probs, dim=1)
        policy_acc = (pred_actions == target_actions).float().mean().item()

        self.network.train()

        val_metrics = {
            "val_loss": loss.item(),
            "val_policy_loss": policy_loss.item(),
            "val_value_loss": value_loss.item(),
            "policy_acc": policy_acc,
        }

        if self.writer:
            self.writer.add_scalar(
                "Loss/val", val_metrics["val_loss"], self.global_step
            )
            self.writer.add_scalar("Policy_Accuracy/val", policy_acc, self.global_step)

        return val_metrics

    def close(self):
        if self.writer:
            self.writer.close()
