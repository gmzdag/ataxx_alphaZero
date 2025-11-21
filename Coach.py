import logging
import os
import sys
import glob
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import mlflow
import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.training_epoch = 0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer, *_ = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                result = []
                for x in trainExamples:
                    board_example, player_who_played, policy, _ = x

                    if player_who_played == self.curPlayer:
                        value = r
                    else:
                        value = -r
                    
                    result.append((board_example, policy, value))
                
                return result
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            mlflow.log_metric('iteration_index', i, step=i)
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                mlflow.log_metric('selfplay_examples', len(iterationTrainExamples), step=i)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # Examples kaydetme opsiyonel (bellek tasarrufu için)
            if getattr(self.args, 'save_examples', True):
                self.saveTrainExamples(i - 1)
            else:
                log.info('Skipping examples save (save_examples=False, bellek tasarrufu)')

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            training_history = self.nnet.train(trainExamples)
            for epoch_stats in training_history:
                self.training_epoch += 1
                mlflow.log_metric('train_loss_pi', epoch_stats['loss_pi'], step=self.training_epoch)
                mlflow.log_metric('train_loss_v', epoch_stats['loss_v'], step=self.training_epoch)
                mlflow.log_metric('train_epoch', epoch_stats['epoch'], step=self.training_epoch)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            mlflow.log_metric('arena_new_wins', nwins, step=i)
            mlflow.log_metric('arena_prev_wins', pwins, step=i)
            mlflow.log_metric('arena_draws', draws, step=i)
            acceptance_rate = float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0.0
            mlflow.log_metric('arena_acceptance_rate', acceptance_rate, step=i)
            if pwins + nwins == 0 or acceptance_rate < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                mlflow.log_metric('model_accepted', 0, step=i)
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                mlflow.log_metric('model_accepted', 1, step=i)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='latest.pth.tar')
                
                # best.pth.tar için examples dosyasını da kaydet
                if getattr(self.args, 'save_examples', True):
                    self.saveBestExamples()
                
                # Eski dosyaları temizle
                self.cleanupOldFiles(i)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def cleanupOldFiles(self, current_iteration):
        """Eski checkpoint ve examples dosyalarını temizle (bellek tasarrufu)"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            return
        
        # Eski checkpoint dosyalarını temizle
        keep_checkpoints = getattr(self.args, 'keep_checkpoints', 3)
        checkpoint_pattern = os.path.join(folder, 'checkpoint_*.pth.tar')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # Checkpoint numaralarına göre sırala
        def get_iteration_num(filename):
            try:
                basename = os.path.basename(filename)
                num_str = basename.replace('checkpoint_', '').replace('.pth.tar', '')
                return int(num_str)
            except:
                return -1
        
        checkpoint_files.sort(key=get_iteration_num)
        
        # Son N checkpoint'i tut, gerisini sil
        if len(checkpoint_files) > keep_checkpoints:
            files_to_delete = checkpoint_files[:-keep_checkpoints]
            for f in files_to_delete:
                try:
                    os.remove(f)
                    log.info(f'Deleted old checkpoint: {os.path.basename(f)}')
                    # İlgili examples dosyasını da sil
                    examples_file = f + '.examples'
                    if os.path.exists(examples_file):
                        os.remove(examples_file)
                        log.info(f'Deleted old examples: {os.path.basename(examples_file)}')
                except Exception as e:
                    log.warning(f'Could not delete {f}: {e}')
        
        # Eski examples dosyalarını temizle (checkpoint'e bağlı olmayan)
        keep_examples = getattr(self.args, 'keep_examples', 5)
        examples_pattern = os.path.join(folder, 'iteration_*.examples')
        examples_files = glob.glob(examples_pattern)
        
        def get_iteration_num_examples(filename):
            try:
                basename = os.path.basename(filename)
                num_str = basename.replace('iteration_', '').replace('.examples', '')
                return int(num_str)
            except:
                return -1
        
        examples_files.sort(key=get_iteration_num_examples)
        
        if len(examples_files) > keep_examples:
            files_to_delete = examples_files[:-keep_examples]
            for f in files_to_delete:
                try:
                    os.remove(f)
                    log.info(f'Deleted old examples file: {os.path.basename(f)}')
                except Exception as e:
                    log.warning(f'Could not delete {f}: {e}')

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
        
        # Eski dosyaları temizle
        self.cleanupOldFiles(iteration)

    def saveBestExamples(self):
        """best.pth.tar için examples dosyasını kaydet"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, 'best.pth.tar.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
        log.info(f'Saved best.pth.tar.examples with {len(self.trainExamplesHistory)} iteration(s) of examples.')

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        # Windows yol birleştirme sorununu çözmek için normalize et
        modelFile = os.path.normpath(modelFile)
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            log.warning('Examples dosyası yok - eğitime sıfırdan başlayacak (ilk iterasyonda self-play yapılacak)')
            # Otomatik devam et (interaktif input yerine)
            self.trainExamplesHistory = []  # Boş başla
            self.skipFirstSelfPlay = False  # İlk iterasyonda self-play yap
            log.info('Continuing without examples file...')
        else:
            log.info("File with trainExamples found. Loading it...")
            try:
                with open(examplesFile, "rb") as f:
                    self.trainExamplesHistory = Unpickler(f).load()
                log.info(f'Loading done! Loaded {len(self.trainExamplesHistory)} iteration(s) of examples.')
                # examples based on the model were already collected (loaded)
                self.skipFirstSelfPlay = True
            except Exception as e:
                log.error(f'Error loading examples file: {e}')
                log.warning('Starting with empty examples history...')
                self.trainExamplesHistory = []
                self.skipFirstSelfPlay = False
