# play.py
import torch
import numpy as np
from game.board import ChineseChessBoard
from model.network import ChessNet, MoveEncoder
from model.mcts import MCTS


def human_vs_ai():
    """人机对战"""
    config = {
        "num_res_blocks": 10,
        "channels": 128,
        "c_puct": 1.5,
        "num_simulations": 800,
        "temperature": 0.1,
    }

    move_encoder = MoveEncoder()
    action_size = move_encoder.action_size

    network = ChessNet(config["num_res_blocks"], config["channels"], action_size)
    checkpoint = torch.load("checkpoints/model_best.pth", map_location="cpu")
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    mcts = MCTS(network, move_encoder, config)
    board = ChineseChessBoard()

    print("中国象棋 AI 对战")
    print("输入格式: 起始行 起始列 目标行 目标列")
    print("例如: 7 4 6 4 (兵向前一步)")

    while True:
        board.display()

        if board.current_player == 1:
            print("\n你的回合 (红方)")
            legal = board.get_legal_moves()

            while True:
                try:
                    inp = input("输入走法: ").strip().split()
                    move = tuple(map(int, inp))
                    if move in legal:
                        break
                    print("非法走法,请重试")
                except:
                    print("输入格式错误")
        else:
            print("\nAI 思考中...")
            probs = mcts.search(board)

            action_idx = np.argmax(probs)
            move = move_encoder.decode(action_idx)

            if move is None:
                legal = board.get_legal_moves()
                move = legal[0] if legal else None

            print(f"AI 走: {move}")

        if move:
            game_over, _ = board.make_move(move)
            if game_over:
                board.display()
                _, winner = board.check_game_over()
                if winner == 1:
                    print("红方(你)获胜!")
                else:
                    print("黑方(AI)获胜!")
                break


if __name__ == "__main__":
    human_vs_ai()
