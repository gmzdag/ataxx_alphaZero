import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ataxx.AtaxxGame import AtaxxGame


def decode_action(n, action):
    return np.unravel_index(action, (n, n, n, n))


def pick_lexicographic_action(game, board, player):
    valids = game.getValidMoves(board, player)
    if not np.any(valids):
        return None
    actions = np.flatnonzero(valids)
    sorted_actions = sorted(actions, key=lambda a: decode_action(game.n, int(a)))
    return int(sorted_actions[0])


def test_deterministic_selfplay_reaches_terminal_state():
    game = AtaxxGame(n=5, timer_limit=20)
    board = game.getInitBoard()
    player = 1
    max_moves = 300

    for _ in range(max_moves):
        result = game.getGameEnded(board, player)
        if result != 0:
            break

        action = pick_lexicographic_action(game, board, player)
        assert action is not None, "Geçerli hamle bekleniyordu"

        board, player, timers = game.getNextState(board, player, action, elapsed=0.1)
        assert timers[1] >= 0 and timers[-1] >= 0
    else:
        pytest.fail("Self-play döngüsü maks hamle sınırı içinde sona ermedi")

    assert result != 0


def test_string_representation_is_stable_across_calls():
    game = AtaxxGame(n=5)
    board = game.getInitBoard()
    rep1 = game.stringRepresentation(board)
    rep2 = game.stringRepresentation(board.copy())

    assert isinstance(rep1, (bytes, bytearray))
    assert rep1 == rep2
