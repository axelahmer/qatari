import torch.nn as nn
from modules.qnet import QNet


class NatureNet(QNet):
    """
    The implementation used in:
    Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." nature 518(7540): 529-533.
    """

    def __init__(self, in_channels, num_actions, writer=None):
        super().__init__(writer)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, x):
        return self.net(x)
