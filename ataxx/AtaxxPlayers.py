"""
Bu dosya, Ataxx oyunu için iki basit oyuncu türünü tanımlar:
- HumanAtaxxPlayer: Kullanıcıdan terminal üzerinden hamle girişi alır.
- RandomAtaxxPlayer: Geçerli hamleler arasından rastgele birini seçer.
"""
import numpy as np
from .AtaxxDisplay import AtaxxDisplay

class HumanAtaxxPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        AtaxxDisplay.display(board)
        n = self.game.n
        print("Hamle girin: 'x y nx ny'")
        while True:
            s = input().strip().lower()
            try:
                x, y, nx, ny = map(int, s.split())
                if 0 <= x < n and 0 <= y < n and 0 <= nx < n and 0 <= ny < n:
                    a = np.ravel_multi_index((x, y, nx, ny), (n, n, n, n))
                    valids = self.game.getValidMoves(board, 1)
                    if valids[a]:
                        return int(a)
            except:
                pass
            print("Geçersiz hamle. Örnek: 2 3 3 4")

class RandomAtaxxPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)  # MCTS wrapper genelde canonical formda çağırır
        aidx = np.flatnonzero(valids)
        if len(aidx) == 0:
            raise ValueError("Geçerli hamle yokken RandomAtaxxPlayer çağrıldı")
        return int(np.random.choice(aidx))
