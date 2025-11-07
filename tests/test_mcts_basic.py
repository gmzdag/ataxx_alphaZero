"""
Ataxx AlphaZero - MCTS Mini Test
----------------------------------
Bu dosya, Ataxx oyun ortamı üzerinde Monte Carlo Tree Search (MCTS) ve 
sinir ağı tahmincisinin (NNetWrapper) birlikte doğru şekilde çalıştığını test eder.

Amaç:
- MCTS’in AlphaZero tarzı politika (π) üretimini test etmek.
- Modelin hamle olasılıklarını (policy vector) düzgün normalize edip etmediğini görmek.
- Entegre sistemin (AtaxxGame + NNetWrapper + MCTS) sağlıklı çalıştığını doğrulamak.

Adım adım işlemler:
1️ Oyun ortamı (7x7 tahta) oluşturulur.
2️ Nöral ağ (NNetWrapper) yüklenir.
3️ MCTS, belirlenen parametrelerle (simülasyon sayısı ve cpuct değeri) başlatılır.
4️ Başlangıç tahtası kanonik forma dönüştürülür.
5️ MCTS üzerinden politika (hamle olasılıkları) hesaplanır ve ilk 20’si ile toplam olasılık yazdırılır.

Bu test, AlphaZero tabanlı Ataxx ajanının karar verme mekanizmasının temel doğrulamasıdır.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import dotdict
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from MCTS import MCTS
import numpy as np

print(" MCTS mini test başlatılıyor...")

game = AtaxxGame(7)
nnet = NNetWrapper(game)
args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
mcts = MCTS(game, nnet, args)

board = game.getInitBoard()
player = 1
canon = game.getCanonicalForm(board, player)

canon_input = np.stack([(canon == 1).astype(np.float32),
                        (canon == -1).astype(np.float32)])

pi = mcts.getActionProb(canon_input, temp=1)
print("\nPolicy uzunluğu:", len(pi))
np.set_printoptions(precision=4, suppress=False)
print("İlk 20 olasılık:", pi[:20])
print("Toplam olasılık:", np.sum(pi))
