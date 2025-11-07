"""
AtaxxGame sınıfı, oyunun temel kurallarını ve mantığını tanımlar.
Tahta başlangıcı, geçerli hamleler, hamle sonrası durum güncellemesi,
oyun bitiş koşulları ve zamanlayıcı yönetimini içerir.
"""
from Game import Game
from .AtaxxLogic import Board
import numpy as np
import time
import copy

class AtaxxGame(Game):
    """
    Ataxx game controller (no pass version)
    1 = player1 (white/red)
    -1 = player2 (black/blue)
    """

    def __init__(self, n=7, timer_limit=100):
        self.n = n
        self.action_size = n * n * n * n  # no +1 for pass
        self.timer_limit = timer_limit
        self.timers = {1: timer_limit, -1: timer_limit}

    # ------------------------------
    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces, copy=True)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.action_size

    # ------------------------------
    def _encode_action(self, x, y, nx, ny):
        return np.ravel_multi_index((x, y, nx, ny), (self.n, self.n, self.n, self.n))

    def _decode_action(self, a):
        return np.unravel_index(a, (self.n, self.n, self.n, self.n))

    # ------------------------------
    def getNextState(self, board, player, action, start_time=None):
        """Return new board, next player, and updated timers"""
        elapsed = (time.time() - start_time) if start_time else 0

        b = Board(self.n)
        b.pieces = np.copy(board)

        # Decode and validate move
        x, y, nx, ny = np.unravel_index(action, (self.n, self.n, self.n, self.n))
        dist = max(abs(nx - x), abs(ny - y))
        if dist not in (1, 2):
            print(f"⚠ Illegal move ignored: {x,y} → {nx,ny} (dist={dist})")
            self.timers[player] -= elapsed
            return b.pieces, -player, copy.deepcopy(self.timers)

        b.execute_move(((x, y), (nx, ny)), player)
        self.timers[player] -= elapsed
        return b.pieces, -player, copy.deepcopy(self.timers)

    # ------------------------------
    def getValidMoves(self, board, player):
        valids = np.zeros(self.action_size, dtype=np.uint8)
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves = b.get_legal_moves(player)

        for (x, y), (nx, ny) in legal_moves:
            idx = self._encode_action(x, y, nx, ny)
            valids[idx] = 1
        return valids

    # ------------------------------
    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)

        # Timer control
        if self.timers[1] <= 0:
            return -1  # player1 timeout → player2 wins
        if self.timers[-1] <= 0:
            return 1   # player2 timeout → player1 wins

        # Full board
        if b.is_full():
            diff = b.countDiff()
            if diff > 0: return 1
            if diff < 0: return -1
            return 1e-4  # draw

        # No pieces
        num_p1 = np.sum(b.pieces == 1)
        num_p2 = np.sum(b.pieces == -1)
        if num_p1 == 0: return -1
        if num_p2 == 0: return 1

        # No legal moves
        no1 = not b.has_legal_moves(1)
        no2 = not b.has_legal_moves(-1)
        if no1 and no2:
            diff = b.countDiff()
            if diff > 0: return 1
            if diff < 0: return -1
            return 1e-4
        if not b.has_legal_moves(player):
            return -1  # pass yok, hamlesi olmayan kaybeder
        return 0

    # ------------------------------
    def getCanonicalForm(self, board, player):
        return board * player

    def stringRepresentation(self, board):
        return board.tobytes()
