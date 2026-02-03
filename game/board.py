# game/board.py
import numpy as np
from typing import List, Tuple, Optional


class ChineseChessBoard:
    """中国象棋棋盘环境"""

    # 棋子编码
    PIECES = {
        "R": 1,
        "N": 2,
        "B": 3,
        "A": 4,
        "K": 5,
        "C": 6,
        "P": 7,  # 红方
        "r": -1,
        "n": -2,
        "b": -3,
        "a": -4,
        "k": -5,
        "c": -6,
        "p": -7,  # 黑方
    }

    # 棋子中文名
    PIECE_NAMES = {
        1: "車",
        2: "馬",
        3: "相",
        4: "仕",
        5: "帥",
        6: "炮",
        7: "兵",
        -1: "车",
        -2: "马",
        -3: "象",
        -4: "士",
        -5: "将",
        -6: "砲",
        -7: "卒",
    }

    def __init__(self):
        self.board = np.zeros((10, 9), dtype=np.int8)
        self.current_player = 1  # 1=红方, -1=黑方
        self.move_history = []
        self.reset()

    def reset(self):
        """重置棋盘到初始状态"""
        self.board = np.zeros((10, 9), dtype=np.int8)

        # 初始化红方 (下方)
        self.board[9] = [1, 2, 3, 4, 5, 4, 3, 2, 1]  # 車馬相仕帥仕相馬車
        self.board[7, [1, 7]] = 6  # 炮
        self.board[6, [0, 2, 4, 6, 8]] = 7  # 兵

        # 初始化黑方 (上方)
        self.board[0] = [-1, -2, -3, -4, -5, -4, -3, -2, -1]
        self.board[2, [1, 7]] = -6
        self.board[3, [0, 2, 4, 6, 8]] = -7

        self.current_player = 1
        self.move_history = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        获取棋盘状态编码 (用于神经网络输入)
        返回: (14, 10, 9) 的张量
        - 前7个通道: 红方各棋子的位置
        - 后7个通道: 黑方各棋子的位置
        """
        state = np.zeros((14, 10, 9), dtype=np.float32)

        for piece_type in range(1, 8):
            # 红方棋子
            state[piece_type - 1] = (self.board == piece_type).astype(np.float32)
            # 黑方棋子
            state[piece_type + 6] = (self.board == -piece_type).astype(np.float32)

        # 如果当前是黑方走棋,翻转视角
        if self.current_player == -1:
            state = state[:, ::-1, ::-1]  # 翻转棋盘
            state = np.concatenate([state[7:14], state[0:7]], axis=0)  # 交换红黑

        return state

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """获取当前玩家所有合法走法"""
        moves = []
        for row in range(10):
            for col in range(9):
                piece = self.board[row, col]
                if piece * self.current_player > 0:  # 是当前玩家的棋子
                    moves.extend(self._get_piece_moves(row, col, piece))
        return moves

    def _get_piece_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """获取指定棋子的所有合法走法"""
        moves = []
        piece_type = abs(piece)

        if piece_type == 1:  # 車
            moves = self._get_rook_moves(row, col, piece)
        elif piece_type == 2:  # 馬
            moves = self._get_knight_moves(row, col, piece)
        elif piece_type == 3:  # 相/象
            moves = self._get_bishop_moves(row, col, piece)
        elif piece_type == 4:  # 仕/士
            moves = self._get_advisor_moves(row, col, piece)
        elif piece_type == 5:  # 帥/将
            moves = self._get_king_moves(row, col, piece)
        elif piece_type == 6:  # 炮
            moves = self._get_cannon_moves(row, col, piece)
        elif piece_type == 7:  # 兵/卒
            moves = self._get_pawn_moves(row, col, piece)

        return moves

    def _get_rook_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """車的走法 - 直线移动"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            for i in range(1, 10):
                nr, nc = row + dr * i, col + dc * i
                if not (0 <= nr < 10 and 0 <= nc < 9):
                    break
                target = self.board[nr, nc]
                if target == 0:
                    moves.append((row, col, nr, nc))
                elif target * piece < 0:  # 敌方棋子
                    moves.append((row, col, nr, nc))
                    break
                else:  # 己方棋子
                    break
        return moves

    def _get_knight_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """馬的走法 - 日字形,需检查蹩马腿"""
        moves = []
        # (行偏移, 列偏移, 蹩马腿位置)
        knight_moves = [
            (-2, -1, -1, 0),
            (-2, 1, -1, 0),
            (2, -1, 1, 0),
            (2, 1, 1, 0),
            (-1, -2, 0, -1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (1, 2, 0, 1),
        ]

        for dr, dc, br, bc in knight_moves:
            nr, nc = row + dr, col + dc
            block_r, block_c = row + br, col + bc

            if not (0 <= nr < 10 and 0 <= nc < 9):
                continue
            if self.board[block_r, block_c] != 0:  # 蹩马腿
                continue

            target = self.board[nr, nc]
            if target * piece <= 0:  # 空位或敌方
                moves.append((row, col, nr, nc))

        return moves

    def _get_bishop_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """相/象的走法 - 田字形,不能过河"""
        moves = []
        # (行偏移, 列偏移, 塞象眼位置)
        bishop_moves = [(-2, -2, -1, -1), (-2, 2, -1, 1), (2, -2, 1, -1), (2, 2, 1, 1)]

        # 相不能过河
        if piece > 0:  # 红相
            valid_rows = range(5, 10)
        else:  # 黑象
            valid_rows = range(0, 5)

        for dr, dc, er, ec in bishop_moves:
            nr, nc = row + dr, col + dc
            eye_r, eye_c = row + er, col + ec

            if not (nr in valid_rows and 0 <= nc < 9):
                continue
            if self.board[eye_r, eye_c] != 0:  # 塞象眼
                continue

            target = self.board[nr, nc]
            if target * piece <= 0:
                moves.append((row, col, nr, nc))

        return moves

    def _get_advisor_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """仕/士的走法 - 斜线移动,限于九宫"""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 九宫范围
        if piece > 0:  # 红仕
            valid_rows = range(7, 10)
        else:  # 黑士
            valid_rows = range(0, 3)
        valid_cols = range(3, 6)

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if nr in valid_rows and nc in valid_cols:
                target = self.board[nr, nc]
                if target * piece <= 0:
                    moves.append((row, col, nr, nc))

        return moves

    def _get_king_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """帥/将的走法 - 直线一步,限于九宫"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        if piece > 0:
            valid_rows = range(7, 10)
        else:
            valid_rows = range(0, 3)
        valid_cols = range(3, 6)

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if nr in valid_rows and nc in valid_cols:
                target = self.board[nr, nc]
                if target * piece <= 0:
                    moves.append((row, col, nr, nc))

        return moves

    def _get_cannon_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """炮的走法 - 移动同車,吃子需要炮架"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            jumped = False
            for i in range(1, 10):
                nr, nc = row + dr * i, col + dc * i
                if not (0 <= nr < 10 and 0 <= nc < 9):
                    break
                target = self.board[nr, nc]

                if not jumped:
                    if target == 0:
                        moves.append((row, col, nr, nc))
                    else:
                        jumped = True  # 找到炮架
                else:
                    if target != 0:
                        if target * piece < 0:  # 敌方棋子
                            moves.append((row, col, nr, nc))
                        break

        return moves

    def _get_pawn_moves(self, row: int, col: int, piece: int) -> List[Tuple]:
        """兵/卒的走法 - 过河前只能前进,过河后可左右"""
        moves = []

        if piece > 0:  # 红兵 (向上)
            forward = (-1, 0)
            crossed = row < 5
        else:  # 黑卒 (向下)
            forward = (1, 0)
            crossed = row > 4

        # 前进
        nr, nc = row + forward[0], col + forward[1]
        if 0 <= nr < 10:
            if self.board[nr, nc] * piece <= 0:
                moves.append((row, col, nr, nc))

        # 过河后可以左右移动
        if crossed:
            for dc in [-1, 1]:
                nc = col + dc
                if 0 <= nc < 9:
                    if self.board[row, nc] * piece <= 0:
                        moves.append((row, col, row, nc))

        return moves

    def make_move(self, move: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        执行走法
        返回: (游戏是否结束, 奖励)
        """
        fr, fc, tr, tc = move

        captured = self.board[tr, tc]
        self.board[tr, tc] = self.board[fr, fc]
        self.board[fr, fc] = 0

        self.move_history.append((move, captured))

        # 检查游戏是否结束
        game_over, winner = self.check_game_over()

        if game_over:
            if winner == self.current_player:
                reward = 1.0
            elif winner == -self.current_player:
                reward = -1.0
            else:
                reward = 0.0  # 平局
        else:
            reward = 0.0

        self.current_player *= -1  # 切换玩家
        return game_over, reward

    def check_game_over(self) -> Tuple[bool, int]:
        """
        检查游戏是否结束
        返回: (是否结束, 获胜方)
        """
        # 检查将帅是否存在
        red_king = np.any(self.board == 5)
        black_king = np.any(self.board == -5)

        if not red_king:
            return True, -1  # 黑方胜
        if not black_king:
            return True, 1  # 红方胜

        # 可以添加更多结束条件: 长将、和棋等
        return False, 0

    def display(self):
        """打印棋盘"""
        print("\n  0 1 2 3 4 5 6 7 8")
        print("  ─────────────────")
        for i, row in enumerate(self.board):
            line = f"{i}│"
            for piece in row:
                if piece == 0:
                    line += "· "
                else:
                    line += self.PIECE_NAMES.get(piece, "?") + " "
            print(line)
        print()
