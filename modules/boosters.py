import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.tensorboard import SummaryWriter


class AdvantageBooster1(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions, writer: SummaryWriter):
        super().__init__()
        self.logging = False
        self.num_actions = num_actions
        self.writer = writer

        # state embedding
        self.state_embed = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        n_low_estimators = 7 * 7
        self.low = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=num_actions, kernel_size=1, stride=1)
        )

        n_med_estimators = 3 * 3
        self.unfold_low = nn.Unfold(kernel_size=3, stride=2)
        self.med = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=num_actions + 3 * 3, kernel_size=1, stride=1)
        )

        self.unfold_med = nn.Unfold(kernel_size=1, stride=1)
        self.high = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions + n_med_estimators)
        )

    def forward(self, x):
        s = self.state_embed(x)

        # TODO replace with narrow
        adv_alpha = self.low(s[:, 0:32, :, :])  # NxAx7x7
        adv_beta, w_beta = th.split(self.med(s[:, 32:52, :, :]), self.num_actions, dim=1)
        # NxAx3x3, N x weights (9) x 3 x 3
        q_gamma, w_gamma = th.split(self.high(s[:, 52:64, :, :]), self.num_actions, dim=1)
        # NxA, N x weights (9)

        # shape and abs weights
        w_beta = w_beta.flatten(2).abs()  # N x weights (9) x blocks (9)
        w_gamma = w_gamma.unsqueeze(1).abs()  # N x weights (1) x blocks (9)

        # weight advantages
        delta_beta = self.weight_tensor(adv_alpha, w_beta, self.unfold_low)  # N x A x blocks (9)
        delta_beta = delta_beta.reshape(x.size(0), self.num_actions, 3, 3)  # N x A x 3 x 3
        delta_gamma = self.weight_tensor(adv_beta + delta_beta, w_gamma, self.unfold_med).sum(2)  # NxA

        # add boosted correction term
        q = q_gamma + delta_gamma

        # if self.logging is True:
        #     self.writer.add_image('obs', x[0])
        #     self.writer.add_histogram('w_high', w_high.reshape(1, 1, 3, 3))

        return q

    def log_forward(self, x):
        self.logging = True
        y = self(x)
        self.logging = False

        return y

    def weight_tensor(self, t, w, unfolder):
        """
        returns the weighted sum over an unfolded tensor
        :param t: NxAxH1xW1 tensor to unflold
        :param w: NxMx(H2xW2) weights
        :param unfolder: nn.Unfold object used to unfold t
        :return: NxAx(H2xH2)
        """
        uf = unfolder(t).reshape(t.size(0), self.num_actions, unfolder.kernel_size ** 2,
                                -1)  # N x A x estimators x blocks
        w = w.unsqueeze(1)  # N x 1 x smalls x blocks : allow broadcast over actions
        wt = uf * w
        ws = wt.sum(2)  # mean across smalls

        # debugging
        # t1 = t[0,:,2:5,4:7]
        # w1 = w[0,0,:,5].reshape(1,3,3)
        # ws1 = (t1*w1).sum(1).sum(1)
        #
        # print(f'me: {ws1}')
        # print(f'ws: {ws[0, :, 5]}')

        return ws  # N x A x bigs
