import torch.nn as nn
import torch.nn.functional as F
import torch as th
from modules.qnet import QNet

class SplitNet2(QNet):
    """

    """

    def __init__(self, in_channels, num_actions, writer=None):
        super().__init__(writer)

        # state embedding
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # latent q embedding
        self.nin1 = nn.Conv2d(66, 32, kernel_size=1, stride=1)
        self.nin2 = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

        # state mixer
        self.lin1 = nn.Linear(64 * 7 * 7, 32 * 7 * 7, bias=True)
        self.lin2 = nn.Linear(32 * 7 * 7, 1 * 7 * 7, bias=True)

        # create position block
        lin = th.arange(0., 7.)
        rows = th.row_stack((lin,) * 7).reshape(1, 1, 7, 7)
        cols = th.column_stack((lin,) * 7).reshape(1, 1, 7, 7)
        self.pos_block = th.cat((rows, cols), dim=1).cuda()/7.

        self.pos_blocks = th.stack((self.pos_block.squeeze(0),) * 32, dim=0).cuda()

    def forward(self, x):
        # embed state
        s = F.relu(self.conv1(x))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))  # N x 64 x 7 x 7

        # calc qs
        if s.shape[0] == 1:
            q = th.cat((s, self.pos_block), dim=1)
        else:
            q = th.cat((s, self.pos_blocks), dim=1)  # N x (64+2=66) x 7 x 7
        q = F.relu(self.nin1(q))
        q = self.nin2(q)
        q = q.flatten(2)
        q = q.permute(0, 2, 1)

        # calc weights
        w = s.flatten(1)
        w = F.relu(self.lin1(w))
        w = self.lin2(w)
        w = w.unsqueeze(-1)

        # weight qs
        qw = q * w
        qw = qw.sum(1)

        return qw
