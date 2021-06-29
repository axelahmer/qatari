import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MixerNet(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        super().__init__()

        # nature nets
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.nin1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.nin2 = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

        self.q_weights = nn.Linear(in_features=64 * 7 * 7, out_features=7 * 7)
        self.relu = nn.ReLU()

    def forward(self, state):
        # embed state
        s = self.relu(self.conv1(state))
        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))

        # calc agent qs
        qs = self.relu(self.nin1(s))
        qs = self.nin2(qs).flatten(2).permute(0, 2, 1)  # N x 49 X A

        # calc weights
        s_flat = s.flatten(1)
        w = self.q_weights(s_flat).unsqueeze(-1)  # N x 49 X 1

        # weight qs
        weighted_qs = w * qs

        return weighted_qs.sum(1)
