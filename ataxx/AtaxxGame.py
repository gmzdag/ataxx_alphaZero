"""
AtaxxGame Modülü
=================
Bu modül, Ataxx oyununun temel kurallarını, oyun durumlarını ve
hamle geçiş mantığını yöneten `AtaxxGame` sınıfını içerir.

Sınıfın Sorumlulukları
----------------------
- Tahta (board) başlatma ve oyun durumunu yönetme.
- Hamlelerin (actions) kodlanması, çözülmesi ve doğrulanması.
- Oyuncuların sıralı hamleleri ve zaman yönetimi.
- Oyun bitiş koşullarının (win / draw / timeout) belirlenmesi.

Tarih: 2025  
"""

from Game import Game
from .AtaxxLogic import Board
import numpy as np


class AtaxxGame(Game):
    """Ataxx oyun denetleyicisi (pass hamlesi olmayan sürüm)."""

    def __init__(self, n=7, timer_limit=100):
        self.n = n
        self.action_size = n * n * n * n
        self.timer_limit = timer_limit
        self.reset_timers()

    def reset_timers(self):
        """Zamanlayıcıları başlangıç değerine sıfırla"""
        self.timers = {1: self.timer_limit, -1: self.timer_limit}

    # ------------------------------
    def getInitBoard(self):
        """Başlangıç tahtasını döndürür."""
        self.reset_timers()  # Her yeni oyunda zamanlayıcıları sıfırla
        b = Board(self.n)
        return np.array(b.pieces, copy=True)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.action_size

    # ------------------------------
    def _encode_action(self, x, y, nx, ny):
        """Bir hamleyi tek indeksli forma dönüştürür."""
        return np.ravel_multi_index((x, y, nx, ny),
                                    (self.n, self.n, self.n, self.n))

    def _decode_action(self, a):
        """Bir indeksin hamle koordinatlarını döndürür."""
        return np.unravel_index(a, (self.n, self.n, self.n, self.n))

    # ------------------------------
    def getNextState(self, board, player, action, start_time=None, elapsed=0.0):
        """
        Verilen hamleyi uygular ve sonucu döndürür.

        Dönüş:
            (new_board, next_player, updated_timers)
        Bu sürümde yalnızca geçerli hamleler oynanabilir.
        Geçersiz bir hamle fonksiyona asla ulaşmaz.
        """
        # Board tipini güvenceye al
        if not isinstance(board, np.ndarray):
            board = np.array(board.pieces, copy=True)

        b = Board(self.n)
        b.pieces = np.copy(board)

        x, y, nx, ny = self._decode_action(action)

        # Süreyi her durumda azalt
        if elapsed and elapsed > 0:
            self.timers[player] -= elapsed

        b.execute_move(((x, y), (nx, ny)), player)
        return b.pieces, -player, dict(self.timers)


    # ------------------------------
    def getValidMoves(self, board, player):
        """Belirtilen oyuncu için geçerli hamleleri (0/1 dizisi) döndürür."""
        valids = np.zeros(self.action_size, dtype=np.uint8)
        b = Board(self.n)
        b.pieces = np.copy(board)
        for (x, y), (nx, ny) in b.get_legal_moves(player):
            valids[self._encode_action(x, y, nx, ny)] = 1
        return valids

    # ------------------------------
    def getGameEnded(self, board, player):
        """
        Oyun bitiş durumunu kontrol eder. 
        
        PASS-FREE ATAXX KURALI:
        Bir oyuncunun hamlesi bittiğinde oyun HEMEN biter ve taş sayısına bakılır.
        
        Dönüş değerleri:
        - pozitif değer: player kazandı
        - negatif değer: player kaybetti
        - 1e-4: Beraberlik
        - 0: Oyun devam ediyor
        
        ÖNEMLİ: Bu fonksiyon 'player' parametresi için çağrılır.
        Sonuç her zaman 'player' perspektifinden verilir.
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        # Zamanlayıcı kontrolleri
        if self.timers[1] <= 0:
            return -1.0 * player  # Player 1 timeout
        if self.timers[-1] <= 0:
            return 1.0 * player   # Player -1 timeout

        # Taş sayıları
        num_p1 = np.sum(b.pieces == 1)
        num_p2 = np.sum(b.pieces == -1)
        
        if num_p1 == 0:
            return -1.0 * player  # Player 1 no pieces
        if num_p2 == 0:
            return 1.0 * player   # Player -1 no pieces

        # Tahta dolu
        if b.is_full():
            diff = b.countDiff()
            if diff > 0:
                return 1.0 * player
            elif diff < 0:
                return -1.0 * player
            else:
                return 1e-4  # Beraberlik

        # PASS-FREE: Hamle kontrolü
        has_moves_p1 = b.has_legal_moves(1)
        has_moves_p2 = b.has_legal_moves(-1)
        
        if not has_moves_p1 or not has_moves_p2:
            diff = b.countDiff()
            if diff > 0:
                return 1.0 * player
            elif diff < 0:
                return -1.0 * player
            else:
                return 1e-4  # Beraberlik
        
        # Oyun devam ediyor - BU SATIRLAR İF BLOĞUNUN DIŞINDA OLMALI!
        return 0

    # ------------------------------
    def getCanonicalForm(self, board, player):
        """Tahtayı oyuncuya göre normalize eder."""
        return board * player

    def stringRepresentation(self, board):
        """Tahtayı byte dizisine dönüştürür (MCTS için)."""
        return board.astype(np.int8).tobytes()

    def getSymmetries(self, board, pi):
        """Simetri dönüşümleri (şu an yalnızca kimlik dönüşümü)."""
        return [(board, pi)]