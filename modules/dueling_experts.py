import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from modules.qnet import QNet
import seaborn as sns


class DuelingAdvantages(QNet):
    """

    """

    def __init__(self, in_channels, num_actions, writer: SummaryWriter):
        super().__init__(writer)
        self.weights = None
        self.fig, self.ax = fig, ax = plt.subplots(figsize=(6, 6))
        self.fig.show()
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

        # store weights for rendering
        self.weights = w

        if self.logging:
            pass
            # fig, ax = plt.subplots()
            # im = ax.imshow(w[0].reshape(7, 7).data.cpu().numpy())
            # self.writer.add_image('i', x[0])
            # self.writer.add_image('w', w.reshape(1, 7, 7), dataformats='CHW')
            # self.writer.add_figure('wfig', fig)

        return q

    def render(self):
        arr = self.weights.detach()[0].cpu().reshape(7, 7).numpy()
        sns.heatmap(arr, ax=self.ax, cbar=False, annot=True, fmt='.2f')




class DuelingTFAS(QNet):
    """
    Thinking fast and slow module:

    2 streams: fast and slow.

    fast: 49 spatial actors each suggest the local action advantages slow: fully connected layers create a baseline
    advantage and state value, in addition decides how to weight all the advantages.

    weighted advantages creates the total advantage and then combine with state value to generate qs, as in a dueling
    network.
    """

    def __init__(self, in_channels, num_actions, writer: SummaryWriter = None):
        super().__init__(writer)

        # embed state
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.fast_state = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.slow_state = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        ### FAST SIDE

        self.adv_actors = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(24, num_actions, kernel_size=1, stride=1),
            nn.Flatten(2)
        )

        ### SLOW SIDE
        state_dim = 32 * 7 * 7

        # value
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # base advantage
        self.adv_base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # adv_weights
        self.adv_weights = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 7 * 7 + 1),
            nn.Softmax()
        )

    def forward(self, x):
        # embed state
        s = self.conv1(x)
        s_fast = self.fast_state(s)
        s_slow = self.slow_state(s).flatten(1)

        # calc advantages
        adv_actors = self.adv_actors(s_fast)  # N x A x M
        adv_base = self.adv_base(s_slow).unsqueeze(-1)  # N x A x 1
        adv_all = th.cat((adv_base, adv_actors), dim=2)  # N x A x (M+1)
        adv_w = self.adv_weights(s_slow).unsqueeze(1)  # N x 1 x (M+1)
        adv = (adv_all * adv_w).sum(2)  # N x A
        adv = adv - adv.mean(1).unsqueeze(-1)  # N x A

        # calc value
        val = self.value(s_slow).expand(adv.shape)  # N X A

        # calc qs
        qs = val + adv  # N x A

        if self.writer is not None:
            self.writer.add_scalars('weights', adv_w[0, 0])
            self.writer.add_image('im', x[0], dataformats='CHW')
            # print(adv_w[0, 0])

        return qs

# class DuelingCommunicatingAdvantages(QNet):
#     def __init__(self):
#         super().__init__()
#         self.num_actions = num_actions
#
#         # nature nets
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4,
#                                padding=(4, 4))  # 22x22
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=(1, 1))  # 11x11
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1))  # 11x11
#
#         self.nin1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.nin2 = nn.Conv2d(64, num_actions + 1, kernel_size=1, stride=1)
#
#         self.num_actors = 11 * 11
#
#         self.relu = nn.ReLU()
#
#     def forward(self, state):
#         # embed state
#
#         s = self.relu(self.conv1(state)).contiguous()
#         s = self.relu(self.conv2(s))
#         s = self.relu(self.conv3(s))
#
#         # calc agent qs
#         qs = self.relu(self.nin1(s))
#         qs = self.nin2(qs).flatten(2)  # N x (A + 1) x self.num_actors
#
#         # calc weights
#         s_flat = s.flatten(1)
#
#         qs = qs.unsqueeze(-1)  # N x (A + 1) X self.num_actors X 1
#
#         qs_n = qs.narrow(1, 0, self.num_actions)  # N x A X self.num_actors X 1
#
#         w = qs.narrow(1, self.num_actions, 1)  # N x 1 X self.num_actors X 1
#         w = F.softmax(w, dim=2)
#
#         res = qs_n.mul(w)  # N x A x self.num_actors X 1
#         res = res.sum(2)  # N x A x 1
#
#         res = res.squeeze(2)
#
#         return res
