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
        print("Hamle girin: 'x y nx ny' ya da 'pass'")
        while True:
            s = input().strip().lower()
            if s == "pass":
                return self.game.getActionSize() - 1
            try:
                x, y, nx, ny = map(int, s.split())
                if 0 <= x < n and 0 <= y < n and 0 <= nx < n and 0 <= ny < n:
                    a = np.ravel_multi_index((x, y, nx, ny), (n, n, n, n))
                    return a
            except:
                pass
            print("Geçersiz format. Örnek: 2 3 3 4  (ya da 'pass')")

class RandomAtaxxPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)  # MCTS wrapper genelde canonical formda çağırır
        aidx = np.flatnonzero(valids)
        return int(np.random.choice(aidx))
