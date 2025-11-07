"""
Ataxx AlphaZero Test Script
----------------------------------
Bu dosya, Ataxx oyunu ortamını ve sinir ağı tahmincisini (NNetWrapper) test etmek için hazırlanmıştır. 
Amaç, hem oyun mantığının (geçerli hamle, yeni durum, oyuncu değişimi, zaman yönetimi) 
hem de AlphaZero tarzı modelin (policy + value tahmini) doğru şekilde çalıştığını doğrulamaktır.

Adım adım yapılan işlemler:
1️ Oyun ortamı oluşturulur (7x7 Ataxx tahtası).
2️ Başlangıç tahtasındaki geçerli hamleler alınır.
3️ İlk geçerli hamle yapılır ve yeni tahta durumu ile kalan süreler görüntülenir.
4️ Nöral ağ modeli (NNetWrapper) yüklenir.
5️ Yeni tahtanın kanonik formu oluşturularak modelden policy (hamle dağılımı) ve value (tahta değeri) tahmini alınır.

Bu script, sistemin temel bileşenlerinin (AtaxxGame + NNetWrapper) entegrasyon testidir.
"""

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper

# 1️ Oyun ortamını oluştur
game = AtaxxGame(n=7)
board = game.getInitBoard()
print("Başlangıç tahtası:\n", board)

# 2️ Geçerli hamleleri al
valid_moves = game.getValidMoves(board, player=1)
print("Geçerli hamle sayısı:", np.sum(valid_moves))

# 3️ İlk geçerli hamleyi yap 
first_move = np.flatnonzero(valid_moves)[0]
start_time = time.time()
new_board, next_player, timers = game.getNextState(board, player=1, action=int(first_move), start_time=start_time)
print("İlk hamle sonrası tahta:\n", new_board)
print("Sıradaki oyuncu:", next_player)
print("Kalan zamanlar:", timers)

# 4️ Nöral ağı oluştur 
nnet = NNetWrapper(game)
print("Model başarıyla yüklendi ✅")

# 5️ Kanonik formu oluştur ve tahmin al
canon = game.getCanonicalForm(new_board, player=next_player)
pi, v = nnet.predict(np.stack([(canon == 1).astype(np.float32),
                               (canon == -1).astype(np.float32)]))
print("\nAğdan gelen policy uzunluğu:", len(pi))
print("Value (durum skoru):", float(v))
