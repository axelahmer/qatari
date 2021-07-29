import torch.nn as nn
import torch.nn.functional as F
from modules.qnet import QNet


class MixerNet(QNet):
    """

    """

    def __init__(self, in_channels, num_actions, writer=None):
        super().__init__(writer)

        # nature nets
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.nin1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.nin2 = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

        self.q_weights = nn.Linear(in_features=64 * 7 * 7, out_features=7 * 7)

    def forward(self, state):
        # embed state
        s = F.relu(self.conv1(state))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))

        # calc agent qs
        qs = F.relu(self.nin1(s))
        qs = self.nin2(qs).flatten(2).permute(0, 2, 1)  # N x 49 X A

        # calc weights
        s_flat = s.flatten(1)
        w = self.q_weights(s_flat).unsqueeze(-1)  # N x 49 X 1

        # weight qs
        weighted_qs = w * qs

        return weighted_qs.sum(1)


class ProcMixerNet(QNet):
    """

    """

    def __init__(self, in_channels, num_actions, writer=None):
        super().__init__(writer)

        self.num_actions = num_actions

        # nature nets
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=(4, 4))  # 17
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1, 1))  # 9
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1))  # 9

        self.nin1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.nin2 = nn.Conv2d(64, num_actions + 1, kernel_size=1, stride=1)

        self.num_actors = 9 * 9

        self.relu = nn.ReLU()

    def forward(self, state):
        # embed state

        s = self.relu(self.conv1(state)).contiguous()
        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))

        # calc agent qs
        qs = self.relu(self.nin1(s))
        qs = self.nin2(qs).flatten(2)  # N x (A + 1) x self.num_actors

        # calc weights
        s_flat = s.flatten(1)

        qs = qs.unsqueeze(-1)  # N x (A + 1) X self.num_actors X 1

        qs_n = qs.narrow(1, 0, self.num_actions)  # N x A X self.num_actors X 1

        w = qs.narrow(1, self.num_actions, 1)  # N x 1 X self.num_actors X 1
        w = F.softmax(w, dim=2)

        res = qs_n.mul(w)  # N x A x self.num_actors X 1
        res = res.sum(2)  # N x A x 1

        res = res.squeeze(2)

        return res
