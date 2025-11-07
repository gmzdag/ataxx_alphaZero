# games/Ataxx/pytorch/NNetWrapper.py
import os
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from utils import *
from NeuralNet import NeuralNet
from .AtaxxNNet import AtaxxNNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,  # 512 çok büyük olabilir 7x7 için
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = AtaxxNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                # --- veri tensörlerine dönüştür ---
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs))

                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # --- forward ---
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # --- backward ---
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

    def predict(self, board):
        """board: np array canonical form (1=player, -1=opponent)"""
        start = time.time()
        board = torch.FloatTensor(board.astype(np.float32))
        if args.cuda:
            board = board.cuda()
        board = board.view(1, 2, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
