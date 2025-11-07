"""
Board sınıfı, Ataxx oyun tahtasının durumunu ve temel kurallarını temsil eder.
Taş yerleşimi, geçerli hamlelerin bulunması, hamle uygulanması 
ve taşların ele geçirilmesi işlemlerini içerir.
"""
import numpy as np

class Board:
    """
    Ataxx board representation (no pass version)
    1 = player1 (white/red)
    -1 = player2 (black/blue)
    0 = empty
    """

    def __init__(self, n=7):
        self.n = n
        self.pieces = np.zeros((n, n), dtype=np.int8)
        self.init_position()

    # -----------------------------
    def init_position(self):
        """Initial 2x2 corners configuration"""
        self.pieces[0, 0] = 1
        self.pieces[self.n - 1, self.n - 1] = 1
        self.pieces[0, self.n - 1] = -1
        self.pieces[self.n - 1, 0] = -1

    def inside(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    # -----------------------------
    def get_legal_moves(self, player):
        """Return list of all possible ((x, y), (nx, ny)) moves"""
        moves = []
        directions = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                      if not (dx == 0 and dy == 0)]
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x, y] == player:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if self.inside(nx, ny) and self.pieces[nx, ny] == 0:
                            moves.append(((x, y), (nx, ny)))
        return moves

    # -----------------------------
    def execute_move(self, move, player):
        """Perform clone or jump move, then capture adjacent opponent pieces"""
        (x, y), (nx, ny) = move
        dist = max(abs(nx - x), abs(ny - y))
        if dist == 1:
            # clone
            self.pieces[nx, ny] = player
        elif dist == 2:
            # jump
            self.pieces[nx, ny] = player
            self.pieces[x, y] = 0
        else:
            raise ValueError("Illegal move distance for Ataxx")

        # capture surrounding opponent pieces
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = nx + dx, ny + dy
                if self.inside(cx, cy) and self.pieces[cx, cy] == -player:
                    self.pieces[cx, cy] = player

    # -----------------------------
    def has_legal_moves(self, player):
        return len(self.get_legal_moves(player)) > 0

    def countDiff(self):
        """Return (player1_pieces - player2_pieces)"""
        return int(np.sum(self.pieces == 1) - np.sum(self.pieces == -1))

    def is_full(self):
        return 0 not in self.pieces

    def game_over(self):
        return self.is_full() or (not self.has_legal_moves(1) and not self.has_legal_moves(-1))
