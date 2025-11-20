import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MCTS import MCTS
from ataxx.AtaxxGame import AtaxxGame
from utils import dotdict


class DummyNet:
    def __init__(self, game):
        self.action_size = game.getActionSize()
        self.calls = 0
        logits = np.linspace(1.0, -1.0, self.action_size, dtype=np.float32)
        self.policy = np.exp(logits) / np.exp(logits).sum()

    def predict(self, board):
        self.calls += 1
        board = np.asarray(board, dtype=np.int8)
        scaled_value = float(np.tanh(board.sum() / 10.0))
        return self.policy.copy(), scaled_value


@pytest.fixture
def mcts_components():
    game = AtaxxGame(n=5)
    net = DummyNet(game)
    args = dotdict({'numMCTSSims': 15, 'cpuct': 1.2})
    return game, net, args


def test_mcts_policy_support_matches_valid_actions(mcts_components):
    game, net, args = mcts_components
    mcts = MCTS(game, net, args)

    board = game.getInitBoard()
    canon = game.getCanonicalForm(board, player=1)
    probs = np.array(mcts.getActionProb(canon, temp=1), dtype=np.float32)
    valids = game.getValidMoves(canon, player=1)

    assert np.all(probs[valids == 0] == 0)
    assert pytest.approx(float(probs.sum()), rel=1e-6) == 1.0
    assert net.calls == args.numMCTSSims


def test_mcts_temp_zero_returns_single_action(mcts_components):
    game, net, args = mcts_components
    mcts = MCTS(game, net, args)

    board = game.getCanonicalForm(game.getInitBoard(), 1)
    probs = np.array(mcts.getActionProb(board, temp=0), dtype=np.float32)

    assert float(probs.sum()) == 1.0
    assert np.count_nonzero(probs) == 1


def test_mcts_respects_terminal_positions(mcts_components):
    game, net, args = mcts_components
    mcts = MCTS(game, net, args)

    board = np.full((game.n, game.n), 1, dtype=np.int8)
    board[0, 0] = -1  # küçük farklarla terminal
    canon = game.getCanonicalForm(board, 1)

    probs = np.array(mcts.getActionProb(canon, temp=1), dtype=np.float32)
    assert np.all(probs == 0)
    assert np.all(game.getValidMoves(canon, 1) == 0)
