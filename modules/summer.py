import torch.nn as nn
import torch.nn.functional as F


class SummerNet(nn.Module):
    """
    conceptually divides observation into 16 partitions, each 21x21x4.
    applies 2 conv layers to the partitions and then 2 linear ('network in network') layers resulting in space: num_actions x4x4
    values are then summed (and averaged) to give q values for each action: num_actions
    """

    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.net1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=7)
        self.net2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.nin1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.nin2 = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.nin1(x))
        x = self.nin2(x)
        x = x.sum(-1).sum(-1) / 16.0
        return x
