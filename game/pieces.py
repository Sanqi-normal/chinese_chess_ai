# game/pieces.py
from enum import Enum
import numpy as np


class PieceType(Enum):
    RED_ROOK = 1  # 車
    RED_KNIGHT = 2  # 馬
    RED_BISHOP = 3  # 相
    RED_ADVISOR = 4  # 仕
    RED_KING = 5  # 帥
    RED_CANNON = 6  # 炮
    RED_PAWN = 7  # 兵

    BLACK_ROOK = -1  # 车
    BLACK_KNIGHT = -2  # 马
    BLACK_BISHOP = -3  # 象
    BLACK_ADVISOR = -4  # 士
    BLACK_KING = -5  # 将
    BLACK_CANNON = -6  # 砲
    BLACK_PAWN = -7  # 卒


class Piece:
    def __init__(self, piece_type: int, row: int, col: int):
        self.piece_type = piece_type
        self.row = row
        self.col = col

    @property
    def is_red(self) -> bool:
        return self.piece_type > 0

    @property
    def name(self) -> str:
        names = {
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
        return names.get(self.piece_type, "?")

    def get_valid_moves(self, board) -> list:
        pass


class Rook(Piece):
    pass


class Knight(Piece):
    pass


class Bishop(Piece):
    pass


class Advisor(Piece):
    pass


class King(Piece):
    pass


class Cannon(Piece):
    pass


class Pawn(Piece):
    pass
