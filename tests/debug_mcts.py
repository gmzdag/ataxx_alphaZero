"""
Training Data Debugging Script
Bu script, Coach'tan gelen training verilerini detaylı şekilde inceler
ve loss_v=0 sorununun kaynağını bulur.
"""

import numpy as np
import sys
import os

# Projenin kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from Coach import Coach
from utils import dotdict


def inspect_episode_data(game, args, num_episodes=3):
    """
    Birkaç episode çalıştır ve üretilen training verisini detaylı incele
    """
    print("\n" + "="*80)
    print("🔍 EPISODE DATA INSPECTION")
    print("="*80)
    
    nnet = NNetWrapper(game)
    coach = Coach(game, nnet, args)
    
    for ep_num in range(num_episodes):
        print(f"\n{'─'*80}")
        print(f"📊 Episode #{ep_num + 1}")
        print(f"{'─'*80}")
        
        # Bir episode çalıştır
        examples = coach.executeEpisode()
        
        print(f"\n✅ Episode completed!")
        print(f"   Total examples collected: {len(examples)}")
        
        # Training examples formatı: (board, pi, v)
        boards = [ex[0] for ex in examples]
        policies = [ex[1] for ex in examples]
        values = [ex[2] for ex in examples]
        
        # VALUE ANALİZİ (EN ÖNEMLİ)
        print(f"\n{'='*60}")
        print(f"📊 VALUE TARGETS ANALYSIS")
        print(f"{'='*60}")
        print(f"   Total values: {len(values)}")
        print(f"   Unique values: {set(values)}")
        print(f"   Value counts:")
        unique_vals, counts = np.unique(values, return_counts=True)
        for val, count in zip(unique_vals, counts):
            percentage = (count / len(values)) * 100
            print(f"      {val:+.4f}: {count:4d} times ({percentage:.1f}%)")
        
        print(f"\n   Statistics:")
        print(f"      Min:  {np.min(values):+.6f}")
        print(f"      Max:  {np.max(values):+.6f}")
        print(f"      Mean: {np.mean(values):+.6f}")
        print(f"      Std:  {np.std(values):.6f}")
        
        # SORUN TESPİTİ
        if len(set(values)) == 1:
            print(f"\n⚠️  CRITICAL PROBLEM DETECTED!")
            print(f"   All values are identical: {values[0]}")
            print(f"   This causes loss_v to be 0!")
            print(f"\n   Possible reasons:")
            print(f"   1. All games end in draw")
            print(f"   2. getGameEnded() returns same value for all positions")
            print(f"   3. Value assignment in executeEpisode() is broken")
        elif np.std(values) < 0.01:
            print(f"\n⚠️  WARNING: Very low variance in values!")
            print(f"   Std: {np.std(values):.6f}")
            print(f"   This will make training difficult")
        else:
            print(f"\n✅ Values look good - sufficient variance detected")
        
        # POLİCY ANALİZİ
        print(f"\n{'='*60}")
        print(f"📊 POLICY ANALYSIS")
        print(f"{'='*60}")
        
        # Policy numpy array mi list mi kontrol et
        first_policy = policies[0]
        if isinstance(first_policy, np.ndarray):
            print(f"   ✅ Policy is numpy array")
            print(f"   Policy shape: {first_policy.shape}")
        elif isinstance(first_policy, list):
            print(f"   ⚠️  Policy is Python list (should be numpy array)")
            first_policy = np.array(first_policy)
            print(f"   Converted shape: {first_policy.shape}")
        
        print(f"   Action space size: {len(first_policy)}")
        
        # İlk 3 policy'yi incele
        print(f"\n   First 3 policies (non-zero entries):")
        for i, p in enumerate(policies[:3]):
            p_arr = np.array(p) if isinstance(p, list) else p
            non_zero = np.where(p_arr > 0.001)[0]
            print(f"      Policy {i}: {len(non_zero)} non-zero actions")
            if len(non_zero) < 10:
                print(f"         Actions: {non_zero}")
        
        # BOARD ANALİZİ
        print(f"\n{'='*60}")
        print(f"📊 BOARD ANALYSIS")
        print(f"{'='*60}")
        print(f"   Board shape: {boards[0].shape}")
        print(f"   First board:")
        print(boards[0])
        print(f"\n   Last board:")
        print(boards[-1])
        
        # Oyun sonu bilgisi
        final_board = boards[-1]
        final_value = values[-1]
        print(f"\n   Final position value: {final_value:+.4f}")
        
        # Taş sayımı
        p1_pieces = np.sum(final_board == 1)
        p2_pieces = np.sum(final_board == -1)
        print(f"   Final piece count:")
        print(f"      Player 1 (white): {p1_pieces}")
        print(f"      Player -1 (black): {p2_pieces}")
        print(f"      Difference: {p1_pieces - p2_pieces:+d}")


def test_game_ended_consistency(game, num_tests=10):
    """
    getGameEnded() fonksiyonunun tutarlı sonuç verip vermediğini test et
    """
    print("\n" + "="*80)
    print("🔬 TESTING getGameEnded() CONSISTENCY")
    print("="*80)
    
    from ataxx.AtaxxLogic import Board
    
    for test_num in range(num_tests):
        print(f"\n{'─'*60}")
        print(f"Test #{test_num + 1}")
        print(f"{'─'*60}")
        
        # Random bir oyun durumu oluştur
        b = Board(game.n)
        
        # Bazı hamleleri simüle et
        num_moves = np.random.randint(5, 20)
        player = 1
        
        for _ in range(num_moves):
            moves = b.get_legal_moves(player)
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            b.execute_move(move, player)
            player = -player
        
        board_state = b.pieces
        
        # Her iki player için getGameEnded() çağır
        result_for_p1 = game.getGameEnded(board_state, 1)
        result_for_p2 = game.getGameEnded(board_state, -1)
        
        print(f"   Board state:")
        print(f"      Player 1 pieces: {np.sum(board_state == 1)}")
        print(f"      Player -1 pieces: {np.sum(board_state == -1)}")
        print(f"      Empty squares: {np.sum(board_state == 0)}")
        
        print(f"\n   getGameEnded() results:")
        print(f"      For player 1:  {result_for_p1:+.4f}")
        print(f"      For player -1: {result_for_p2:+.4f}")
        
        # Tutarlılık kontrolü
        if result_for_p1 != 0 and result_for_p2 != 0:
            # Oyun bitti - değerler ters işaretli olmalı
            if abs(result_for_p1 + result_for_p2) > 0.001:
                print(f"\n   ⚠️  WARNING: Inconsistent results!")
                print(f"      Expected: result_for_p1 = -result_for_p2")
                print(f"      Got: {result_for_p1} and {result_for_p2}")
            else:
                print(f"   ✅ Results are consistent (opposite signs)")


def test_value_assignment_logic():
    """
    Coach.executeEpisode() içindeki value assignment mantığını test et
    """
    print("\n" + "="*80)
    print("🔬 TESTING VALUE ASSIGNMENT LOGIC")
    print("="*80)
    
    print("\nSimulating different game outcomes:\n")
    
    scenarios = [
        {"final_player": 1, "game_result": 1.0, "name": "Player 1 wins"},
        {"final_player": 1, "game_result": -1.0, "name": "Player 1 loses"},
        {"final_player": -1, "game_result": 1.0, "name": "Player -1 wins"},
        {"final_player": -1, "game_result": -1.0, "name": "Player -1 loses"},
        {"final_player": 1, "game_result": 0.0, "name": "Draw (player 1 last)"},
    ]
    
    # Örnek episode datası: (board, player_who_played, policy, value_placeholder)
    mock_episode = [
        (None, 1, None, None),    # Player 1 played
        (None, -1, None, None),   # Player -1 played
        (None, 1, None, None),    # Player 1 played
        (None, -1, None, None),   # Player -1 played
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"   Final player (curPlayer): {scenario['final_player']}")
        print(f"   Game result (from getGameEnded): {scenario['game_result']:+.4f}")
        print(f"{'─'*60}")
        
        final_player = scenario['final_player']
        r = scenario['game_result']
        
        print(f"\n   Assigned values to each position:")
        for i, (_, player_who_played, _, _) in enumerate(mock_episode):
            # Coach.executeEpisode() mantığı
            if player_who_played == final_player:
                value = r
            else:
                value = -r
            
            print(f"      Move {i+1} (played by {player_who_played:+d}): value = {value:+.4f}")
        
        # Unique values kontrolü
        values = []
        for _, player_who_played, _, _ in mock_episode:
            if player_who_played == final_player:
                values.append(r)
            else:
                values.append(-r)
        
        unique_values = set(values)
        print(f"\n   Unique values in episode: {unique_values}")
        
        if len(unique_values) == 1:
            print(f"   ⚠️  All values are identical! This is the problem!")
        else:
            print(f"   ✅ Values have variance - good!")


def main():
    """Ana test fonksiyonu"""
    
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE TRAINING DATA DEBUG")
    print("="*80)
    
    # Minimal args
    args = dotdict({
        'numMCTSSims': 10,  # Az simülasyon - hızlı test
        'tempThreshold': 5,
        'cpuct': 1.0,
    })
    
    game = AtaxxGame(n=7)
    
    # TEST 1: Value assignment mantığı
    print("\n" + "█"*80)
    print("TEST 1: VALUE ASSIGNMENT LOGIC")
    print("█"*80)
    test_value_assignment_logic()
    
    # TEST 2: getGameEnded tutarlılığı
    print("\n" + "█"*80)
    print("TEST 2: getGameEnded() CONSISTENCY")
    print("█"*80)
    test_game_ended_consistency(game, num_tests=5)
    
    # TEST 3: Gerçek episode verisi
    print("\n" + "█"*80)
    print("TEST 3: REAL EPISODE DATA")
    print("█"*80)
    inspect_episode_data(game, args, num_episodes=3)
    
    print("\n" + "="*80)
    print("✅ DEBUG COMPLETE")
    print("="*80)
    print("\nLook for these warning signs:")
    print("  ⚠️  All values are identical")
    print("  ⚠️  Very low variance in values")
    print("  ⚠️  Inconsistent getGameEnded() results")
    print("\nIf you see any of these, that's the root cause of loss_v=0")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()