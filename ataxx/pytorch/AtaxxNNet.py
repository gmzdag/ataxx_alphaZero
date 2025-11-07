import torch
import torch.nn as nn
import torch.nn.functional as F

"""
2 giriş kanalı: mevcut oyuncunun taşları (1), rakibinki (-1).
CNN böylece “hangi tarafta kim var” ayrımını öğreniyor.

3 konv katmanı: 7×7 olduğu için fazla derinliğe gerek yok.

F.log_softmax(pi, dim=1) → log-olasılık verir (AlphaZero formatı).

tanh(v) → değer tahmini -1 ↔ +1 aralığında.
"""
class AtaxxNNet(nn.Module):
    def __init__(self, game, args):
        super(AtaxxNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()  # 7x7
        self.action_size = game.getActionSize()           # 2402
        self.args = args

        # 2 giriş kanalı: player ve opponent
        self.conv1 = nn.Conv2d(2, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        # Flatten sonrası tam bağlı katmanlar
        conv_output_size = args.num_channels * (self.board_x - 2) * (self.board_y - 2)
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Policy & Value çıkışları
        self.fc_pi = nn.Linear(512, self.action_size)
        self.fc_v = nn.Linear(512, 1)

    def forward(self, s):
        # s: (batch, 2, 7, 7)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))

        s = s.view(s.size(0), -1)  # flatten

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        pi = self.fc_pi(s)                  # policy logits
        v = torch.tanh(self.fc_v(s))        # value ∈ [-1,1]
        return F.log_softmax(pi, dim=1), v
