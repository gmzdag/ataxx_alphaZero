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
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board, pi, v) or (board, player, pi, v)
        """
        # Examples formatını normalize et: (board, pi, v) formatına çevir
        normalized_examples = []
        for ex in examples:
            if len(ex) == 3:
                # Zaten doğru format: (board, pi, v)
                normalized_examples.append(ex)
            elif len(ex) == 4:
                # Eski format: (board, player, pi, v) -> (board, pi, v)
                board, player, pi, v = ex
                normalized_examples.append((board, pi, v))
            else:
                raise ValueError(f"Invalid example format: expected 3 or 4 elements, got {len(ex)}: {ex}")
        
        examples = normalized_examples
        
        epoch_history = []
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = max(1, int(np.ceil(len(examples) / args.batch_size)))
            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                # boards: (B, n, n) kanonik → (B, 2, n, n)
                boards_np = np.array(boards, dtype=np.float32)                     # (B, n, n)
                boards_np = np.stack([(boards_np == 1).astype(np.float32),
                                    (boards_np == -1).astype(np.float32)], axis=1)  # (B, 2, n, n)

                boards     = torch.FloatTensor(boards_np)
                target_pis = torch.FloatTensor(np.array(target_pis, dtype=np.float32))  # (B, action_size), 1-hot/dağılım
                
                target_vs_np = np.array(target_vs, dtype=np.float32)
                assert np.all(np.abs(target_vs_np) <= 1.1), f"Invalid target values: {target_vs_np}"
                target_vs = torch.FloatTensor(target_vs_np)

                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # --- forward ---
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # --- backward ---
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

            epoch_history.append({
                'epoch': epoch + 1,
                'loss_pi': pi_losses.avg,
                'loss_v': v_losses.avg,
            })
        return epoch_history

    def predict(self, board):
        """board: np array canonical form (n x n), values in {-1,0,1} w.r.t current player"""
        import torch
        import numpy as np
        self.nnet.eval()

        # 7x7 kanonik tahtayı 2 kanala çevir: [mevcut oyuncu, rakip]
        # shape: [1, 2, n, n]
        b = board.astype(np.float32)
        two_channel = np.stack([(b == 1).astype(np.float32),
                                (b == -1).astype(np.float32)], axis=0)[None, ...]
        x = torch.from_numpy(two_channel)
        if args.cuda:
            x = x.cuda()

        with torch.no_grad():
            pi_log, v = self.nnet(x)  # pi_log: [1, action_size] (log-prob ya da logits), v: [1, 1]

        # Eğer model log-softmax döndürüyorsa exp ile olasılığa çeviriyoruz (senin koddaki gibi)
        pi = torch.exp(pi_log).cpu().numpy()[0]   # [action_size]
        v  = v.squeeze().item()
        return pi, v


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', save_optimizer=None):
        """
        Checkpoint kaydet. Optimizer state opsiyonel (bellek tasarrufu için).
        save_optimizer: None ise args.save_optimizer'a bakar, True/False ise direkt kullanır.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        
        # Optimizer state kaydetme kararı
        if save_optimizer is None:
            # args objesinde save_optimizer varsa kullan, yoksa varsayılan True
            save_optimizer = args.get('save_optimizer', True)
        
        checkpoint_dict = {
            'state_dict': self.nnet.state_dict(),
        }
        
        if save_optimizer:
            checkpoint_dict['optimizer'] = self.optimizer.state_dict()
        
        torch.save(checkpoint_dict, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if args.cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
