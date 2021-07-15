import torch.nn as nn
import torch.nn.functional as F
import torch as th


class DuelingAdvantages(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.num_actions = num_actions

        # state embedding
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # advantage experts
        self.a_nin1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.a_nin2 = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

        # advantage expert weights (how much to weight each expert's opinion)
        self.a_lin1 = nn.Linear(64 * 7 * 7, 512)
        self.a_lin2 = nn.Linear(512, 1 * 7 * 7)

        # value layers
        self.v_lin1 = nn.Linear(64 * 7 * 7, 512)
        self.v_lin2 = nn.Linear(512, 1)

    def forward(self, x):
        # embed state
        s = F.relu(self.conv1(x))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))  # N x 64 x 7 x 7
        s_flat = s.flatten(1)

        # calc expert advantage estimates
        a = F.relu(self.a_nin1(s))
        a = self.a_nin2(a)
        a = a.flatten(2).permute(0, 2, 1)

        # calc advantage weights
        w = F.relu(self.a_lin1(s_flat))
        w = self.a_lin2(w)
        w = w.unsqueeze(-1)

        # weight advantage
        a = a * w
        a = a.sum(1)

        # calc value estimate
        v = F.relu(self.v_lin1(s_flat))
        v = self.v_lin2(v)
        v = v.expand(x.size(0), self.num_actions)

        # calculate q estimates
        q = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)

        return q
