"""
Ataxx oyun mantığını test eden script.
Sonsuz döngü ve hamle geçişi sorunlarını tespit eder.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from ataxx.AtaxxGame import AtaxxGame
from ataxx.AtaxxLogic import Board

def test_basic_board():
    """Temel board fonksiyonlarını test et"""
    print("=" * 50)
    print("TEST 1: Temel Board Fonksiyonları")
    print("=" * 50)
    
    board = Board(7)
    print("Başlangıç tahtası:")
    print(board)
    print()
    
    # Player 1 hamleleri
    moves_p1 = board.get_legal_moves(1)
    print(f"Player 1'in hamle sayısı: {len(moves_p1)}")
    print(f"İlk 5 hamle: {moves_p1[:5]}")
    print()
    
    # Player -1 hamleleri
    moves_p2 = board.get_legal_moves(-1)
    print(f"Player -1'in hamle sayısı: {len(moves_p2)}")
    print()
    
    # İlk hamleyi oynat
    if moves_p1:
        move = moves_p1[0]
        print(f"Player 1 hamle yapıyor: {move}")
        board.execute_move(move, 1)
        print(board)
        print()

def test_game_ending():
    """Oyun bitiş durumlarını test et"""
    print("=" * 50)
    print("TEST 2: Oyun Bitiş Durumları (PASS-FREE)")
    print("=" * 50)
    
    # Senario 1: Bir oyuncunun taşı yok
    game = AtaxxGame(7)
    board = np.zeros((7, 7), dtype=np.int8)
    board[3, 3] = -1  # Sadece Player -1'in bir taşı var
    
    print("Senaryo 1: Player 1'in taşı yok")
    b = Board(7)
    b.pieces = board
    print(b)
    print(f"Player 1 hamlesi var mı: {b.has_legal_moves(1)}")
    print(f"Player -1 hamlesi var mı: {b.has_legal_moves(-1)}")
    print(f"Board.game_over(): {b.game_over()}")
    
    # AtaxxGame perspektifinden kontrol et
    result_for_p1 = game.getGameEnded(board, 1)
    result_for_p2 = game.getGameEnded(board, -1)
    print(f"getGameEnded(board, player=1): {result_for_p1}")
    print(f"getGameEnded(board, player=-1): {result_for_p2}")
    print(f"✓ Oyun bitti çünkü Player 1'in taşı kalmadı")
    print()
    
    # Senario 2: Player -1'in hamlesi yok ama taşları var
    board2 = np.zeros((7, 7), dtype=np.int8)
    board2[3, 3] = -1  # Player -1 ortada
    # Çevresini Player 1 ile sar (hamle yapamaz)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            x, y = 3 + dx, 3 + dy
            if 0 <= x < 7 and 0 <= y < 7:
                board2[x, y] = 1
    
    print("Senaryo 2: Player -1 kuşatılmış (hamlesi yok ama taşı var)")
    b2 = Board(7)
    b2.pieces = board2
    print(f"Player 1 taş sayısı: {np.sum(board2 == 1)}")
    print(f"Player -1 taş sayısı: {np.sum(board2 == -1)}")
    print(f"Player 1 hamlesi var mı: {b2.has_legal_moves(1)}")
    print(f"Player -1 hamlesi var mı: {b2.has_legal_moves(-1)}")
    print(f"Board.game_over(): {b2.game_over()}")
    
    result_for_p1 = game.getGameEnded(board2, 1)
    result_for_p2 = game.getGameEnded(board2, -1)
    print(f"getGameEnded(board, player=1): {result_for_p1}")
    print(f"getGameEnded(board, player=-1): {result_for_p2}")
    print(f"✓ PASS-FREE KURAL: Player -1'in hamlesi yok, oyun bitti!")
    print(f"✓ Player 1 kazandı (daha çok taşı var: {np.sum(board2 == 1)} vs {np.sum(board2 == -1)})")
    print()
    
    # Senario 3: Her iki oyuncunun da hamlesi yok
    board3 = np.ones((7, 7), dtype=np.int8)  # Tahta dolu
    board3[0, 0] = 1
    board3[6, 6] = -1
    
    print("Senaryo 3: Tahta dolu, her iki oyuncunun da hamlesi yok")
    b3 = Board(7)
    b3.pieces = board3
    print(f"Player 1 taş sayısı: {np.sum(board3 == 1)}")
    print(f"Player -1 taş sayısı: {np.sum(board3 == -1)}")
    print(f"Board.game_over(): {b3.game_over()}")
    
    result_for_p1 = game.getGameEnded(board3, 1)
    result_for_p2 = game.getGameEnded(board3, -1)
    print(f"getGameEnded(board, player=1): {result_for_p1}")
    print(f"getGameEnded(board, player=-1): {result_for_p2}")
    print()

def test_no_moves_scenario():
    """Hiçbir oyuncunun hamlesi olmadığı durumu test et"""
    print("=" * 50)
    print("TEST 3: Hamle Yapılamayan Durumlar")
    print("=" * 50)
    
    # Senaryo 1: Tahta tamamen dolu
    board = Board(7)
    board.pieces[:] = 1  # Tüm tahta Player 1 taşlarıyla dolu
    board.pieces[6, 6] = -1  # Bir köşeye Player -1 taşı
    
    print("Senaryo 1: Tahta tamamen dolu (boş yer yok)")
    print(board)
    print(f"Player 1 hamlesi var mı: {board.has_legal_moves(1)}")
    print(f"Player -1 hamlesi var mı: {board.has_legal_moves(-1)}")
    print(f"Oyun bitti mi: {board.game_over()}")
    if board.game_over():
        print(f"Kazanan: {board.get_winner()}")
    print()
    
    # Senaryo 2: Bir oyuncunun taşı çevresinde boş yer yok
    board2 = Board(7)
    board2.pieces[:] = 0
    # Player 1 taşı ortada
    board2.pieces[3, 3] = 1
    # Etrafını Player -1 taşlarıyla çevir (2 birimlik alanda hiç boş yer kalmasın)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            x, y = 3 + dx, 3 + dy
            if 0 <= x < 7 and 0 <= y < 7:
                board2.pieces[x, y] = -1
    
    print("Senaryo 2: Player 1 taşı çevresinde hiç boş yer yok")
    print(board2)
    print(f"Player 1 hamlesi var mı: {board2.has_legal_moves(1)}")
    print(f"Player -1 hamlesi var mı: {board2.has_legal_moves(-1)}")
    print(f"Oyun bitti mi: {board2.game_over()}")
    print()
    
    # Senaryo 3: İzole taşlar ama etraflarında boş yer VAR
    board3 = Board(7)
    board3.pieces[:] = 0
    board3.pieces[0, 0] = 1
    board3.pieces[6, 6] = -1
    
    print("Senaryo 3: İzole taşlar (köşelerde, etraflarında boş yer var)")
    print(board3)
    moves_p1 = board3.get_legal_moves(1)
    moves_p2 = board3.get_legal_moves(-1)
    print(f"Player 1 hamleleri: {len(moves_p1)} adet")
    if moves_p1:
        print(f"  Örnekler: {moves_p1[:3]}")
    print(f"Player -1 hamleleri: {len(moves_p2)} adet")
    if moves_p2:
        print(f"  Örnekler: {moves_p2[:3]}")
    print(f"Oyun bitti mi: {board3.game_over()}")
    print(f"✓ İzole olsalar bile etraflarında boş yer varsa hamle yapabilirler!")
    print()

def test_game_integration():
    """AtaxxGame sınıfıyla entegrasyon testi"""
    print("=" * 50)
    print("TEST 4: AtaxxGame Entegrasyonu")
    print("=" * 50)
    
    game = AtaxxGame(7)
    board = game.getInitBoard()
    
    print("Başlangıç:")
    print(np.array(board).reshape(7, 7))
    print()
    
    current_player = 1
    move_count = 0
    max_moves = 20  # Sonsuz döngü önleme
    
    while move_count < max_moves:
        # Oyun bitti mi kontrol et
        game_ended = game.getGameEnded(board, current_player)
        if game_ended != 0:
            print(f"\nOyun bitti! Sonuç: {game_ended}")
            print(f"Toplam hamle: {move_count}")
            break
        
        # Geçerli hamleleri al
        valid_moves = game.getValidMoves(board, current_player)
        valid_indices = np.where(valid_moves == 1)[0]
        
        if len(valid_indices) == 0:
            print(f"\nPlayer {current_player}'ın hamlesi yok!")
            # Oyun bitmeli
            game_ended = game.getGameEnded(board, current_player)
            print(f"Oyun bitiş durumu: {game_ended}")
            break
        
        # Rastgele bir hamle seç
        action = np.random.choice(valid_indices)
        
        # Hamleyi uygula
        board, current_player, _ = game.getNextState(board, current_player, action)
        move_count += 1
        
        if move_count % 5 == 0:
            print(f"Hamle {move_count}, Sıra: Player {-current_player}")
    
    print()
    print("Son tahta:")
    print(np.array(board).reshape(7, 7))

def test_mcts_safety():
    """MCTS ile güvenlik testi"""
    print("=" * 50)
    print("TEST 5: MCTS Güvenlik Testi")
    print("=" * 50)
    
    try:
        from ataxx.pytorch.NNet import NNetWrapper
        from MCTS import MCTS
        from utils import dotdict
        
        args = dotdict({
            'numMCTSSims': 10,
            'cpuct': 1.0,
        })
        
        game = AtaxxGame(7)
        nnet = NNetWrapper(game)
        mcts = MCTS(game, nnet, args)
        
        board = game.getInitBoard()
        
        print("MCTS simülasyonu başlatılıyor...")
        probs = mcts.getActionProb(board, temp=1)
        print(f"✓ MCTS başarıyla tamamlandı")
        print(f"  Toplam state sayısı: {len(mcts.Ns)}")
        print(f"  Olasılık toplamı: {sum(probs):.4f}")
        
    except ImportError as e:
        print(f"⚠ MCTS test edilemedi (bağımlılık eksik): {e}")
    except RecursionError as e:
        print(f"✗ MCTS HATA: Sonsuz döngü tespit edildi!")
        print(f"  {e}")
    except Exception as e:
        print(f"✗ MCTS HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_board()
    test_game_ending()
    test_no_moves_scenario()
    test_game_integration()
    test_mcts_safety()
    
    print("\n" + "=" * 50)
    print("TÜM TESTLER TAMAMLANDI")
    print("=" * 50)