# config.py - 参考 ChineseChess-AlphaZero 的配置
config = {
    # --- 网络 ---
    "num_res_blocks": 7,
    "channels": 256,
    # --- MCTS ---
    "c_puct": 1.5,
    "num_simulations": 100,
    "temperature": 1.0,
    "temperature_decay": 0.95,
    "leaf_batch_size": 16,
    "virtual_loss": 3.0,
    "dirichlet_alpha": 0.2,
    "noise_eps": 0.15,
    "tau_decay_rate": 0.9,
    "resign_threshold": -0.98,
    "min_resign_turn": 80,
    "enable_resign_rate": 0.5,
    "max_game_length": 256,
    # --- 奖惩 ---
    "use_material_reward": True,
    "win_reward": 1.0,
    "loss_reward": -1.0,
    "draw_reward": 0.0,
    "material_scale": 0.01,
    # 棋子分值
    "piece_values": {
        1: 100,  # 車
        2: 45,  # 馬
        3: 20,  # 相
        4: 15,  # 仕
        5: 0,  # 帥 (将)
        6: 45,  # 炮
        7: 10,  # 兵
        -1: -100,  # 車
        -2: -45,  # 馬
        -3: -20,  # 象
        -4: -15,  # 士
        -5: 0,  # 将
        -6: -45,  # 炮
        -7: -10,  # 卒
    },
    # --- 训练 ---
    "learning_rate": 0.01,
    "lr_schedule": [
        (0, 0.01),
        (50000, 0.003),
        (150000, 0.0001),
    ],
    "momentum": 0.9,
    "l2_reg": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "buffer_size": 500000,
    "num_iterations": 200,
    "games_per_iteration": 8,
    "train_steps_per_iteration": 100,
    # --- 验证 ---
    "validation_split": 0.02,
    # --- 日志 ---
    "log_dir": "./logs",
    "save_model_steps": 25,
    "log_game_details": True,
    # --- 并行 ---
    "num_self_play_workers": 4,
}
