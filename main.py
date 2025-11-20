import logging
import time

import coloredlogs
import mlflow

from Coach import Coach
from ataxx.AtaxxGame import AtaxxGame as Game
from ataxx.pytorch.NNet import NNetWrapper as nn
from ataxx.pytorch.NNet import args as nnet_args
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    experiment_name = 'ataxx_alpha_zero'
    mlflow.set_experiment(experiment_name)
    run_name = f"ataxx_run_{time.strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        log.info('Starting MLflow run "%s" under experiment "%s"', run_name, experiment_name)
        mlflow.log_params(dict(args))
        mlflow.log_params({f"nnet_{k}": v for k, v in dict(nnet_args).items()})

        log.info('Loading %s...', Game.__name__)
        g = Game(n=7, timer_limit=100)

        log.info('Loading %s...', nn.__name__)
        nnet = nn(g)

        if args.load_model:
            log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        c = Coach(g, nnet, args)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        c.learn()


if __name__ == "__main__":
    main()
