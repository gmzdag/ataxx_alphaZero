import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ataxx.AtaxxGame import AtaxxGame


def decode_action(n, action):
    return np.unravel_index(action, (n, n, n, n))


def encode_action(n, move):
    (x, y), (nx, ny) = move
    return np.ravel_multi_index((x, y, nx, ny), (n, n, n, n))


def count_pieces(board, value):
    return int(np.sum(board == value))


def test_initial_state_has_expected_valid_move_count():
    game = AtaxxGame(n=7)
    board = game.getInitBoard()

    valids = game.getValidMoves(board, player=1)

    assert valids.shape == (game.action_size,)
    assert int(valids.sum()) == 16  # 4 köşe taşı için 4 hamle


def test_clone_and_jump_behaviour_affect_board_and_player_turn():
    game = AtaxxGame(n=7)
    board = game.getInitBoard()
    valids = game.getValidMoves(board, player=1)
    initial_count = count_pieces(board, 1)

    clone_action = next(
        action for action in np.flatnonzero(valids)
        if max(abs(a - b) for a, b in zip(decode_action(game.n, action)[:2],
                                          decode_action(game.n, action)[2:])) == 1
    )
    jump_action = next(
        action for action in np.flatnonzero(valids)
        if max(abs(a - b) for a, b in zip(decode_action(game.n, action)[:2],
                                          decode_action(game.n, action)[2:])) == 2
    )

    clone_board, next_player, _ = game.getNextState(board, player=1, action=int(clone_action))
    cx, cy, cnx, cny = decode_action(game.n, clone_action)
    assert next_player == -1
    assert count_pieces(clone_board, 1) == initial_count + 1
    assert clone_board[cx, cy] == 1 and clone_board[cnx, cny] == 1

    jump_board, _, _ = game.getNextState(board, player=1, action=int(jump_action))
    jx, jy, jnx, jny = decode_action(game.n, jump_action)
    assert count_pieces(jump_board, 1) == initial_count
    assert jump_board[jx, jy] == 0 and jump_board[jnx, jny] == 1


def test_move_converts_adjacent_opponents_and_updates_timer():
    game = AtaxxGame(n=5, timer_limit=10)
    board = np.zeros((5, 5), dtype=np.int8)
    board[2, 2] = 1
    board[2, 3] = -1
    board[3, 2] = -1

    action = encode_action(game.n, ((2, 2), (3, 3)))
    new_board, _, timers = game.getNextState(board, player=1, action=int(action), elapsed=1.25)

    assert new_board[3, 3] == 1
    assert new_board[2, 3] == 1 and new_board[3, 2] == 1
    assert pytest.approx(timers[1], rel=1e-6) == 8.75
    assert timers[-1] == game.timer_limit


def test_game_end_detects_player_without_moves_and_uses_piece_diff():
    game = AtaxxGame(n=5)
    board = np.array(
        [
            [0, -1, -1, 1, -1],
            [-1, -1, -1, -1, 1],
            [-1, -1, -1, -1, 1],
            [-1, 1, -1, 1, 1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    result = game.getGameEnded(board, player=1)
    assert result == -1.0


def test_timeout_detection_returns_correct_winner():
    game = AtaxxGame(n=5, timer_limit=5)
    board = game.getInitBoard()

    game.timers[1] = -0.1
    assert game.getGameEnded(board, player=1) == -1.0

    game.reset_timers()
    game.timers[-1] = -0.1
    assert game.getGameEnded(board, player=1) == 1.0
