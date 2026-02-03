# game/rules.py
from .board import ChineseChessBoard


class GameRules:
    """游戏规则验证器"""

    @staticmethod
    def is_valid_move(board: ChineseChessBoard, move) -> bool:
        legal_moves = board.get_legal_moves()
        return move in legal_moves

    @staticmethod
    def is_check(board: ChineseChessBoard, player: int) -> bool:
        king_pos = GameRules._find_king(board, player)
        if king_pos is None:
            return False

        opponent = -player
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece * opponent > 0:
                    moves = board._get_piece_moves(row, col, piece)
                    for fr, fc, tr, tc in moves:
                        if (tr, tc) == king_pos:
                            return True
        return False

    @staticmethod
    def is_checkmate(board: ChineseChessBoard, player: int) -> bool:
        if not GameRules.is_check(board, player):
            return False

        legal_moves = board.get_legal_moves()
        for move in legal_moves:
            temp_board = board.board.copy()
            fr, fc, tr, tc = move
            temp_board[tr, tc] = temp_board[fr, fc]
            temp_board[fr, fc] = 0

            temp_board_obj = ChineseChessBoard()
            temp_board_obj.board = temp_board
            temp_board_obj.current_player = -player

            if not GameRules.is_check(temp_board_obj, player):
                return False
        return True

    @staticmethod
    def _find_king(board: ChineseChessBoard, player: int) -> tuple:
        king_value = 5 if player > 0 else -5
        for row in range(10):
            for col in range(9):
                if board.board[row, col] == king_value:
                    return (row, col)
        return None

    @staticmethod
    def is_repetition(board: ChineseChessBoard) -> bool:
        if len(board.move_history) < 4:
            return False
        recent_moves = board.move_history[-4:]
        if len(recent_moves) >= 8:
            if recent_moves[-4:] == board.move_history[-8:-4]:
                return True
        return False
