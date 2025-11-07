"""
Ataxx Random Play Test (No-Pass Version)
----------------------------------
Bu dosya, pass hakkı bulunmayan Ataxx oyun ortamının (AtaxxGame) temel işlevlerinin doğru çalıştığını test eder.
Amaç, iki oyuncunun rastgele hamlelerle oyunu tamamlayabildiği dinamik bir oyun döngüsünü gözlemlemektir.

Test Edilen Bileşenler:
- getInitBoard()  →  Başlangıç tahtasının oluşturulması
- getValidMoves() →  Geçerli hamlelerin hesaplanması
- getNextState()  →  Hamle sonrası yeni tahtanın, sıradaki oyuncunun ve sürelerin güncellenmesi
- getGameEnded()  →  Oyun bitiş koşullarının kontrolü

Adım adım işlemler:
1. 7x7 boyutlu Ataxx tahtası ve 100 saniyelik süre limitiyle oyun başlatılır.  
2. Her turda geçerli hamleler belirlenir.  
3. Geçerli hamleler arasından rastgele biri seçilerek uygulanır.  
4. Yeni durum, oyuncu değişimi ve süre bilgileri ekrana yazdırılır.  
5. Oyun, biri kazanana, süre dolana veya maksimum hamle (200) limitine ulaşılana kadar devam eder.

Notlar:
- Pass hakkı yoktur. Hamle yapamayan oyuncu oyunu kaybeder.
- Her oyuncunun kendi süresi vardır; süre sırası geldiğinde azalır.
"""

import sys, os, time
import numpy as np

# Üst klasörü import yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ataxx.AtaxxGame import AtaxxGame

# Oyun başlat
game = AtaxxGame(n=7, timer_limit=100)
board = game.getInitBoard()
player = 1
move_count = 0
MAX_MOVES = 200

print("🎮 Başlangıç durumu:\n", board)

while True:
    # Geçerli hamleleri bul
    valids = game.getValidMoves(board, player)
    valid_indices = np.flatnonzero(valids)

    # Oyun bitti mi?
    result = game.getGameEnded(board, player)
    if result != 0 or len(valid_indices) == 0:
        print("\nOyun bitti! 🎯 (result =", result, ")")
        print("Son tahta:\n", board)
        print("Kalan zamanlar:", game.timers)
        break

    # Rastgele geçerli hamle seç
    start_time = time.time()
    action = np.random.choice(valid_indices)
    board, player, timers = game.getNextState(board, player, int(action), start_time=start_time)
    move_count += 1

    # Durumu yazdır
    print("-" * 30)
    print(f"{move_count}. hamle sonrası tahta (şu an oynayacak: player {player}):")
    print(board)
    print(f"⏱ Süreler -> P1: {timers[1]:.2f}s | P2: {timers[-1]:.2f}s")

    # Limit kontrolü
    if move_count >= MAX_MOVES:
        print("\n⚠ Maksimum hamle sayısına ulaşıldı, oyun durduruldu.")
        break

print("Toplam hamle:", move_count)
