# config.py
config = {
    # --- 网络 ---
    "num_res_blocks": 4,
    "channels": 128,
    # --- MCTS ---
    "c_puct": 1.0,
    "num_simulations": 60,
    "temperature": 0.8,
    "leaf_batch_size": 64,
    "virtual_loss": 2.0,
    # --- 训练 ---
    "learning_rate": 0.001,
    "batch_size": 256,
    "buffer_size": 500000,
    "num_iterations": 200,
    "games_per_iteration": 8,
    "train_steps_per_iteration": 100,
    # --- 并行 ---
    "num_self_play_workers": 4,
}
