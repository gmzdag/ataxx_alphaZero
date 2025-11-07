"""
Ataxx AlphaZero Self-Play Test
----------------------------------
Bu dosya, Ataxx oyunu üzerinde AlphaZero tarzı iki ajan arasında zaman sınırlı bir self-play (kendi kendine oyun) simülasyonu gerçekleştirir. 
Amaç, MCTS (Monte Carlo Tree Search) algoritmasının iki oyuncu arasında dönüşümlü olarak çalıştığı ve 
her oyuncunun süresinin azaldığı bir tam oyun akışını gözlemlemektir.

Test Edilen Bileşenler:
- AtaxxGame      → Tahta yapısı, hamle geçerliliği, oyuncu geçişi, süre kontrolü
- NNetWrapper    → Sinir ağının (policy + value) tahmin mekanizması
- MCTS           → Hamle olasılıklarını (π) üreten arama algoritması
- Timer sistemi  → Oyuncu sürelerinin güncel tutulması ve bitiş koşullarını etkileyebilmesi

Adım adım işlemler:
1️ 7x7 boyutlu tahta ve 100 saniyelik süre limitiyle oyun başlatılır.  
2️ Her oyuncu kendi MCTS ajanını (mcts1, mcts2) kullanarak hamle olasılıklarını üretir.  
3️ MCTS çıktısına göre olasılıksal bir hamle seçilir ve uygulanır.  
4️ Tahta güncellenir, oyuncu değişir ve süreler azaltılır.  
5️ Oyun; biri kazanana, süresi dolana veya maksimum hamle (200) limitine ulaşılana kadar devam eder.

Bu test, AlphaZero tabanlı Ataxx ajanlarının zaman baskısı altında karar verme ve rekabet etme süreçlerini gözlemlemek için kullanılır.
"""
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils import dotdict
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from MCTS import MCTS

print("🤖 Ataxx Self-Play (Timerlı) başlıyor...\n")

game = AtaxxGame(7)
nnet = NNetWrapper(game)

args = dotdict({'numMCTSSims': 10, 'cpuct': 1.0})
mcts1 = MCTS(game, nnet, args)
mcts2 = MCTS(game, nnet, args)

board = game.getInitBoard()
player = 1
timers = {1: 100, -1: 100}
move_count = 0

def display_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    for row in board:
        print(" ".join(symbols[int(x)] for x in row))
    print()

while True:
    move_count += 1
    print(f"\n========= Hamle {move_count} =========")
    display_board(board)
    print(f"⏱ P1: {timers[1]:.1f}s | P2: {timers[-1]:.1f}s")

    canon = game.getCanonicalForm(board, player)
    canon_input = np.stack([(canon == 1).astype(np.float32),
                            (canon == -1).astype(np.float32)])
    mcts = mcts1 if player == 1 else mcts2
    pi = mcts.getActionProb(canon_input, temp=1)
    action = np.random.choice(len(pi), p=pi)

    start_time = time.time()
    board, player, timers = game.getNextState(board, player, action, start_time=start_time)
    result = game.getGameEnded(board, player, timers)

    if result != 0:
        print("\n🏁 Oyun bitti!")
        display_board(board)
        print(f"Sonuç: {result} | P1 süresi: {timers[1]:.2f}s | P2 süresi: {timers[-1]:.2f}s")
        break

    if move_count >= 200:
        print("\n⚠️ Maksimum hamleye ulaşıldı, oyun durduruldu.")
        display_board(board)
        break
