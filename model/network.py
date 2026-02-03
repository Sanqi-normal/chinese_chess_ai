# model/network.py - 优化版
"""
优化:
1. 加 torch.compile() 编译 forward（PyTorch 2.0+），自动融合算子，
   对 ResBlock 栈尤其有效（可 2-3x 加速 inference）
2. 架构不变，保持与原版 checkpoint 兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    __slots__ = ()  # 减少实例字典开销

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """
    中国象棋神经网络 (AlphaZero 架构)
    输入: (batch, 14, 10, 9)
    输出: (log_policy, value)
    """

    def __init__(self, num_res_blocks=5, channels=256, action_size=2086):
        super().__init__()
        self.action_size = action_size

        self.conv_input = nn.Conv2d(14, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 10 * 9, action_size)

        # Value head
        self.value_conv = nn.Conv2d(channels, 4, 1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def compile_network(network):
    """
    用 torch.compile 编译网络 (PyTorch >= 2.0)。
    编译后第一次调用会 JIT 编译（耗几秒），之后推理+训练都快。
    mode="reduce-overhead" 适合反复调用 small batch。
    """
    if not hasattr(torch, "compile"):
        print("⚠ torch.compile 不可用 (PyTorch < 2.0)，使用原版网络")
        return network

    try:
        compiled = torch.compile(network, mode="reduce-overhead", dynamic=False)
        print("✓ torch.compile 编译成功")
        return compiled
    except Exception as e:
        error_str = str(e)
        if "Triton" in error_str or "triton" in error_str.lower():
            print("⚠ Triton 不可用，尝试使用 AOT Eager 后端...")
            try:
                compiled = torch.compile(
                    network, mode="reduce-overhead", dynamic=False, backend="aot_eager"
                )
                print("✓ torch.compile + aot_eager 编译成功")
                return compiled
            except Exception as e2:
                print(f"⚠ AOT Eager 也失败 ({e2})，使用原版网络")
                return network
        else:
            print(f"⚠ torch.compile 不可用 ({e})，使用原版网络")
            return network


class MoveEncoder:
    """走法编码器"""

    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_table()

    def _build_move_table(self):
        idx = 0
        for fr in range(10):
            for fc in range(9):
                for tr in range(10):
                    for tc in range(9):
                        if (fr, fc) != (tr, tc) and self._is_possible_move(
                            fr, fc, tr, tc
                        ):
                            move = (fr, fc, tr, tc)
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        self.action_size = idx
        print(f"总共 {idx} 种可能走法")

    @staticmethod
    def _is_possible_move(fr, fc, tr, tc):
        dr, dc = abs(tr - fr), abs(tc - fc)
        if dr == 0 or dc == 0:  # 車炮
            return True
        if (dr == 2 and dc == 1) or (dr == 1 and dc == 2):  # 馬
            return True
        if dr == 2 and dc == 2:  # 相象
            return True
        if dr == 1 and dc == 1:  # 仕士/将帅
            return True
        return False

    def encode(self, move):
        return self.move_to_idx.get(move, -1)

    def decode(self, idx):
        return self.idx_to_move.get(idx, None)
