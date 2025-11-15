"""
Ataxx AlphaZero Self-Play 
-------------------------------------------------
Bu sürümde oyuncuların süreleri, sıraları geldiğinde akmaya başlar.
Her oyuncunun kendi turunda MCTS düşünme süresi kendi saatinden düşülür.
Oyun, biri kazanırsa, hamle kalmazsa veya süresi dolarsa sona erer.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys, os, time
import numpy as np
from utils import dotdict
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from MCTS import MCTS

print("🤖 Ataxx Self-Play  başlıyor...\n")

# ------------------------------------------------------------------
# 1️⃣ Oyun başlat
game = AtaxxGame(n=7)
nnet = NNetWrapper(game)
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})

mcts1 = MCTS(game, nnet, args)
mcts2 = MCTS(game, nnet, args)

board = game.getInitBoard()
player = 1
timers = {1: 100.0, -1: 100.0}
move_count = 0
max_moves = 200

# ------------------------------------------------------------------
def display_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    for row in board:
        print(" ".join(symbols[int(x)] for x in row))
    print()

# ------------------------------------------------------------------
while True:
    move_count += 1
    print(f"\n========= Hamle {move_count} =========")
    display_board(board)
    print(f"⏱ P1: {game.timers[1]:.2f}s | P2: {game.timers[-1]:.2f}s") 
    turn_start = time.time()

    canon = game.getCanonicalForm(board, player)             
    mcts = mcts1 if player == 1 else mcts2                 
    temp = 1  # veya (1 if move_count < 15 else 0)
    pi = mcts.getActionProb(canon, temp=temp)

    valids = game.getValidMoves(board, player)              
    if np.sum(valids) == 0:
        print(f"♟️ Oyuncu {player} için hamle yok → kayıp.")
        result = -player
        break

    pi = pi * valids
    s = np.sum(pi)
    pi = (pi / s) if s > 0 else (valids / np.sum(valids))    

    action = np.random.choice(len(pi), p=pi)

    elapsed = time.time() - turn_start
    board, next_player, timers = game.getNextState(board, player, action, elapsed=elapsed)
    if next_player == player:
        # yani aynı oyuncuya tekrar sıra geldiyse
        continue
    else:
        player = next_player

    
    result = game.getGameEnded(board, player)
    if result != 0:
        print("\n🏁 Oyun bitti!")
        display_board(board)
        print(f"Sonuç: {result} | P1: {game.timers[1]:.2f}s | P2: {game.timers[-1]:.2f}s")
        break

    #  Oyun bitti mi kontrol et
    result = game.getGameEnded(board, player)
    if result != 0:
        print("\n🏁 Oyun bitti!")
        display_board(board)
        print(f"Sonuç: {result} | P1: {timers[1]:.2f}s | P2: {timers[-1]:.2f}s")
        break

    if move_count >= max_moves:
        print("\n⚠️ Maksimum hamle sayısına ulaşıldı.")
        break
