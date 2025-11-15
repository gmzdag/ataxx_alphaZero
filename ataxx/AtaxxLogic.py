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
        """Check if coordinates are inside the board"""
        # Güvenli tip dönüşümü
        try:
            x = int(x)
            y = int(y)
        except (ValueError, TypeError):
            return False
        return 0 <= x < self.n and 0 <= y < self.n

    # -----------------------------
    def get_legal_moves(self, player):
        """
        Return list of all possible ((x, y), (nx, ny)) moves for the given player.
        
        Ataxx kuralları:
        - Distance 1: Clone (taş çoğalır, eski pozisyon kalır)
        - Distance 2: Jump (taş atlar, eski pozisyon boşalır)
        - Distance > 2: Illegal
        
        Returns:
            list: Liste of ((x, y), (nx, ny)) tuples
        """
        moves = []
        
        # Sadece geçerli hamle mesafeleri: 1 veya 2
        directions = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                dist = max(abs(dx), abs(dy))
                if dist <= 2:  # Sadece 1 veya 2 uzaklık
                    directions.append((dx, dy))
        
        # Oyuncunun her taşı için
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
        """
        Perform clone or jump move, then capture adjacent opponent pieces.
        
        Args:
            move: ((x, y), (nx, ny)) tuple
            player: 1 or -1
        
        Raises:
            ValueError: If move is illegal
        """
        (x, y), (nx, ny) = move
        
        # Hamle mesafesi kontrolü (Chebyshev distance)
        dist = max(abs(nx - x), abs(ny - y))
        
        # Kaynak pozisyon kontrolü
        if self.pieces[x, y] != player:
            raise ValueError(f"No piece at source position ({x}, {y}) for player {player}")
        
        # Hedef pozisyon kontrolü
        if not self.inside(nx, ny):
            raise ValueError(f"Target position ({nx}, {ny}) is outside board")
        
        if self.pieces[nx, ny] != 0:
            raise ValueError(f"Target position ({nx}, {ny}) is not empty")
        
        # Hamle uygula
        if dist == 1:
            # Clone: taş çoğalır
            self.pieces[nx, ny] = player
        elif dist == 2:
            # Jump: taş atlar
            self.pieces[nx, ny] = player
            self.pieces[x, y] = 0
        else:
            raise ValueError(f"Illegal move distance {dist} for Ataxx (must be 1 or 2)")

        # Çevredeki rakip taşları ele geçir (8 komşu)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = nx + dx, ny + dy
                if self.inside(cx, cy) and self.pieces[cx, cy] == -player:
                    self.pieces[cx, cy] = player

    # -----------------------------
    def has_legal_moves(self, player):
        """
        Check if player has any legal moves.
        
        Optimizasyon: Hamle bulunur bulunmaz True döner.
        
        Returns:
            bool: True if player has at least one legal move
        """
        # Erken çıkış optimizasyonu
        directions = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                dist = max(abs(dx), abs(dy))
                if dist <= 2:
                    directions.append((dx, dy))
        
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x, y] == player:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if self.inside(nx, ny) and self.pieces[nx, ny] == 0:
                            return True  # En az bir hamle var
        
        return False

    def countDiff(self):
        """
        Return the difference in piece count (player1 - player2).
        
        Returns:
            int: Positive if player 1 is ahead, negative if player -1 is ahead
        """
        return int(np.sum(self.pieces == 1) - np.sum(self.pieces == -1))

    def is_full(self):
        """Check if the board is completely filled"""
        return not np.any(self.pieces == 0)

    def game_over(self):
        """
        Check if the game is over.
        
        PASS-FREE ATAXX RULE:
        Game ends when ANY of these conditions is met:
        1. Board is full
        2. One or both players have no legal moves
        3. One player has no pieces left
        
        Returns:
            bool: True if game is over
        """
        if self.is_full():
            return True
        
        # Bir oyuncunun taşı kalmadıysa oyun biter
        num_p1 = np.sum(self.pieces == 1)
        num_p2 = np.sum(self.pieces == -1)
        if num_p1 == 0 or num_p2 == 0:
            return True
        
        # PASS-FREE KURAL: Herhangi bir oyuncunun hamlesi yoksa oyun biter
        if not self.has_legal_moves(1) or not self.has_legal_moves(-1):
            return True
        
        return False
    
    def get_winner(self):
        """
        Determine the winner based on piece count.
        Should only be called when game is over.
        
        Returns:
            1: Player 1 wins
            -1: Player -1 wins
            0: Draw
        """
        diff = self.countDiff()
        if diff > 0:
            return 1
        elif diff < 0:
            return -1
        else:
            return 0
    
    def copy(self):
        """Create a deep copy of the board"""
        new_board = Board(self.n)
        new_board.pieces = np.copy(self.pieces)
        return new_board
    
    def __str__(self):
        """String representation for debugging"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        lines = []
        for row in self.pieces:
            lines.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(lines)