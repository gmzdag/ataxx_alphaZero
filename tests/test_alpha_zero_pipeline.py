import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Arena import Arena
from Coach import Coach
from MCTS import MCTS
from ataxx.AtaxxGame import AtaxxGame
from ataxx.pytorch.NNet import NNetWrapper
from utils import dotdict


def first_valid_player_factory(game):
    def player(board):
        valids = game.getValidMoves(board, 1)
        valid_idxs = np.flatnonzero(valids)
        assert len(valid_idxs) > 0, "Hamlesiz durumla karşılaşıldı"
        return int(valid_idxs[0])

    return player


def mcts_player_factory(game, nnet, args):
    mcts = MCTS(game, nnet, args)

    def player(board):
        probs = mcts.getActionProb(board, temp=0)
        return int(np.argmax(probs))

    return player


def test_alpha_zero_smoke_pipeline(tmp_path):
    game = AtaxxGame(n=5, timer_limit=30)
    nnet = NNetWrapper(game)
    coach_args = dotdict({'numMCTSSims': 4, 'cpuct': 1.0, 'tempThreshold': 3})
    coach = Coach(game, nnet, coach_args)

    train_examples = coach.executeEpisode()
    assert len(train_examples) > 0

    for board, policy, value in train_examples:
        assert board.shape == (game.n, game.n)
        assert len(policy) == game.getActionSize()
        assert pytest.approx(float(np.sum(policy)), rel=1e-6) == 1.0
        assert -1.0 <= float(value) <= 1.0

    arena_args = dotdict({'numMCTSSims': 6, 'cpuct': 1.0})
    alpha_player = mcts_player_factory(game, nnet, arena_args)
    baseline_player = first_valid_player_factory(game)
    arena = Arena(alpha_player, baseline_player, game)

    result = arena.playGame(verbose=False)
    if result not in (-1, 1):
        assert pytest.approx(result, rel=1e-6) == 1e-4

