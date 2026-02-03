# Chinese Chess AI - 中国象棋AI

基于 AlphaGo Zero/AlphaZero 算法的中国象棋人工智能程序。

## 功能特点

- **自我对弈训练**：无需人类棋谱，通过自我对弈学习
- **人机对战**：支持人类玩家与AI对弈
- **高性能优化**：
  - PyTorch 2.0 `torch.compile` 编译加速
  - 批量 MCTS 推理，GPU 利用率 >80%
  - 预分配 ReplayBuffer，比 deque 快 3-5x
  - 多进程并行对弈

## 项目结构

```
chinese_chess_ai/
├── game/
│   ├── board.py      # 棋盘逻辑与走法规则
│   ├── pieces.py     # 棋子定义
│   └── rules.py      # 游戏规则
├── model/
│   ├── network.py    # 神经网络架构 (AlphaZero风格)
│   └── mcts.py       # MCTS搜索算法 (批量推理优化)
├── training/
│   ├── self_play.py  # 自我对弈模块
│   └── trainer.py    # 训练模块
├── train.py          # 训练入口
├── play.py           # 人机对战入口
├── config.py         # 配置文件
└── checkpoints/      # 模型保存目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练模型

```bash
python train.py
```

训练配置（可在 `config.py` 中修改）：
- `num_iterations`: 训练轮数
- `games_per_iteration`: 每轮自我对弈局数
- `num_simulations`: MCTS 模拟次数
- `num_res_blocks`: ResNet 残差块数量
- `channels`: 网络通道数

### 人机对战

训练完成后运行：

```bash
python play.py
```

输入格式：`起始行 起始列 目标行 目标列`
例如：`7 4 6 4`（兵向前一步）

## 算法原理

### 神经网络架构
- 输入：14×10×9 的棋盘状态张量（7个通道表示红方棋子，7个通道表示黑方棋子）
- 输出：
  - **策略头**：预测每步棋的概率分布 (2086 种走法)
  - **价值头**：评估当前局面的胜率 (-1 ~ 1)

### MCTS 搜索
1. **选择**：使用 PUCT 公式选择最优子节点
2. **扩展**：创建新节点
3. **评估**：神经网络评估叶节点
4. **回传**：更新路径上所有节点的统计信息

### 训练流程
1. **自我对弈**：使用当前策略进行对弈，收集 (状态, MCTS策略, 奖励) 三元组
2. **训练**：最小化策略损失（交叉熵）和价值损失（MSE）
3. **评估**：定期保存检查点

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_res_blocks | 4 | 残差块数量 |
| channels | 128 | 网络通道数 |
| num_simulations | 60 | MCTS 模拟次数 |
| temperature | 0.8 | 探索温度 |
| learning_rate | 0.001 | 学习率 |
| batch_size | 256 | 批次大小 |
| buffer_size | 500000 | 回放缓冲区大小 |
| num_self_play_workers | 4 | 并行对弈进程数 |

## 硬件要求

- Python 3.8+
- PyTorch 1.9+ (推荐 2.0+ 以获得编译加速)
- CUDA 可选，但强烈推荐（训练速度提升显著）
- 内存 4GB+

## License

MIT License
