# training/trainer.py - 优化版
"""
优化:
1. ReplayBuffer 用 numpy array 预分配 + 随机索引采样，避免 deque 逐项访问
2. 采样后直接 pin_memory -> to(device, non_blocking=True)，掩盖数据传输
3. AMP 统一写法 (torch.amp)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading


class ReplayBuffer:
    """
    预分配 numpy buffer，比 deque 逐元素访问快 3-5x。
    """

    def __init__(self, capacity: int, action_size: int, state_shape: tuple):
        self.capacity = capacity
        self.action_size = action_size
        self.state_shape = state_shape

        # 预分配
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.probs = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

        self.size = 0  # 当前有效数据量
        self.ptr = 0  # 写入指针（环形）
        self.lock = threading.Lock()

    def push(self, data: list):
        """data: [(state, mcts_probs, reward), ...]"""
        with self.lock:
            for state, prob, value in data:
                self.states[self.ptr] = state
                self.probs[self.ptr] = prob
                self.values[self.ptr, 0] = value
                self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + len(data), self.capacity)

    def sample(self, batch_size: int):
        with self.lock:
            indices = np.random.randint(0, self.size, size=batch_size)
            # numpy 花式索引会复制，避免 race condition
            s = self.states[indices].copy()
            p = self.probs[indices].copy()
            v = self.values[indices].copy()

        # pin_memory 让后续 .to(device) 用异步 DMA
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

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.buffer = ReplayBuffer(
            capacity=config.get("buffer_size", 500000),
            action_size=action_size,
            state_shape=state_shape,
        )
        self.batch_size = config.get("batch_size", 512)

        # AMP scaler - disabled for now (testing fp32 speed)
        self.use_amp = False
        self.scaler = None
        print(f"✓ AMP disabled (using fp32)")

    # -------------------------------------------------------- 数据接口
    def add_game_data(self, data: list):
        self.buffer.push(data)

    # --------------------------------------------------------- 训练步
    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        states, target_probs, target_values = self.buffer.sample(self.batch_size)

        # non_blocking: CPU->GPU 异步，CPU 线程不等待传输完成
        states = states.to(self.device, non_blocking=True)
        target_probs = target_probs.to(self.device, non_blocking=True)
        target_values = target_values.to(self.device, non_blocking=True)

        if self.use_amp:
            with torch.autocast(device_type="cuda"):
                pred_log_probs, pred_values = self.network(states)
                policy_loss = -torch.mean(
                    torch.sum(target_probs * pred_log_probs, dim=1)
                )
                value_loss = nn.functional.mse_loss(pred_values, target_values)
                loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero 快
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred_log_probs, pred_values = self.network(states)
            policy_loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))
            value_loss = nn.functional.mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    # ------------------------------------------------------ 多步训练糖
    def train_epoch(self, num_steps: int) -> dict:
        """连续训练 num_steps 步，返回平均 metrics"""
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        actual = 0
        for _ in range(num_steps):
            m = self.train_step()
            if m["loss"] > 0:
                for k in totals:
                    totals[k] += m[k]
                actual += 1
        if actual > 0:
            totals = {k: v / actual for k, v in totals.items()}
        return totals
