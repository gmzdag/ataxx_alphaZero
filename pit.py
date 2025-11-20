import Arena
import numpy as np

from MCTS import MCTS
from ataxx.AtaxxGame import AtaxxGame
from ataxx.AtaxxPlayers import HumanAtaxxPlayer, RandomAtaxxPlayer
from ataxx.AtaxxDisplay import AtaxxDisplay
from ataxx.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict

"""
Bu betik, Ataxx için iki ajanı veya insan vs ajan senaryosunu karşılaştırmak
için kullanılabilir.
"""

BOARD_SIZE = 7
TIMER_LIMIT = 100
human_vs_cpu = True
use_trained_model = False  # Eğitilmiş checkpoint yüklemek için True yapın
checkpoint_path = ('./pretrained_models/ataxx/pytorch/', 'best.pth.tar')
num_games = 2

g = AtaxxGame(n=BOARD_SIZE, timer_limit=TIMER_LIMIT)

if use_trained_model:
    nnet = NNet(g)
    nnet.load_checkpoint(*checkpoint_path)
    args = dotdict({'numMCTSSims': 75, 'cpuct': 1.0})
    mcts = MCTS(g, nnet, args)
    player1 = lambda x: int(np.argmax(mcts.getActionProb(x, temp=0)))
else:
    # Eğitim yapılmadıysa basit rastgele oyuncu kullan
    player1 = RandomAtaxxPlayer(g).play

if human_vs_cpu:
    player2 = HumanAtaxxPlayer(g).play
else:
    player2 = RandomAtaxxPlayer(g).play

arena = Arena.Arena(
    player1,
    player2,
    g,
    display=AtaxxDisplay.display,
    use_timers=human_vs_cpu,
)
print(arena.playGames(num_games, verbose=True))
