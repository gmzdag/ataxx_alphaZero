import logging
import math

import numpy as np

EPS = 1e-8
MAX_SEARCH_DEPTH = 500  # Maksimum rekürsif derinlik güvenlik önlemi

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            valids = self.Vs.get(s)
            if valids is None:
                valids = self.game.getValidMoves(canonicalBoard, 1)
            if np.sum(valids) == 0:
                # no valid moves — game ended
                return valids.astype(np.float32)
            # uniform distribution over valids only
            probs = valids / np.sum(valids)
            return probs.astype(np.float32)
        
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, depth=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Args:
            canonicalBoard: current board state
            depth: current recursion depth (for safety)

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        # Güvenlik kontrolü: Maksimum derinlik
        if depth > MAX_SEARCH_DEPTH:
            log.warning(f"MCTS search exceeded max depth {MAX_SEARCH_DEPTH}")
            # Terminal olmayan bir durumda bile değerlendirme yap
            _, v = self.nnet.predict(canonicalBoard)
            return -v

        s = self.game.stringRepresentation(canonicalBoard)

        # Terminal durumu kontrol et
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # Yeni durum - neural network değerlendirmesi
        if s not in self.Ps:
            pi, v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            
            # Geçerli hamle yoksa - bu bir terminal durumu olmalı
            if np.sum(valids) == 0:
                # Oyun bitmiş, tekrar kontrol et
                game_end = self.game.getGameEnded(canonicalBoard, 1)
                self.Es[s] = game_end
                return -game_end
            
            pi = pi * valids
            ssum = np.sum(pi)
            self.Ps[s] = pi/ssum if ssum > 0 else valids/np.sum(valids)
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        
        # Geçerli hamle yok - terminal durumu kontrol et
        if np.sum(valids) == 0:
            game_end = self.game.getGameEnded(canonicalBoard, 1)
            self.Es[s] = game_end
            return -game_end
        
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player, *_ = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, depth + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v