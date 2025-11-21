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
    'numEps': 60,               # Number of complete self-play games (optimized: 100->60, daha hızlı ama yeterli)
    'tempThreshold': 15,        # Temperature threshold for exploration
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 50000,     # Number of game examples (optimized: 200000->50000, bellek tasarrufu)
    'numMCTSSims': 30,          # Number of games moves for MCTS (optimized: 50->30, daha hızlı)
    'arenaCompare': 20,         # Number of games to play during arena (optimized: 40->20, daha hızlı test)
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,         # True yaparak mevcut checkpoint'i yükle
    'load_folder_file': ('./temp', 'best.pth.tar'),  # best.pth.tar veya latest.pth.tar kullanabilirsin
    'numItersForTrainExamplesHistory': 10,  # Optimized: 20->10, daha az geçmiş tutar (bellek tasarrufu)
    
    # Bellek optimizasyonu için ek parametreler
    'keep_checkpoints': 3,      # Sadece son 3 checkpoint'i tut (eski olanları sil)
    'keep_examples': 5,         # Sadece son 5 examples dosyasını tut
    'save_optimizer': False,    # Optimizer state kaydetme (bellek tasarrufu, eğitim sırasında reset olur)
    'save_examples': True,      # Examples dosyalarını kaydet (False yaparsan bellek tasarrufu, ama eğitime devam ederken examples yoksa sıfırdan başlar)
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