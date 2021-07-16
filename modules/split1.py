import torch.nn as nn
import torch.nn.functional as F
import torch as th


class SplitNet1(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        super().__init__()

        # state embedding A
        self.conv1l = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2l = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3l = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # state embedding B
        # self.conv1r = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2r = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3r = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1)

        # latent q embedding
        self.nin1l = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.nin2l = nn.Conv2d(32, num_actions, kernel_size=1, stride=1)

        self.lin1r = nn.Linear(32 * 7 * 7, 16 * 7 * 7, bias=True)
        self.lin2r = nn.Linear(16 * 7 * 7, 1 * 7 * 7, bias=True)

    def forward(self, x):
        # left and right: spatial state embeddings
        xl = F.relu(self.conv1l(x))
        with th.no_grad():
            xr = xl.clone()
        xr = F.relu(self.conv2r(xr))
        xl = F.relu(self.conv2l(xl))
        with th.no_grad():
            xl_clone = xl.clone()
        xr = th.cat((xr, xl_clone), 1)
        xr = F.relu(self.conv3r(xr))
        xl = F.relu(self.conv3l(xl))

        # left: spatial latent q embeddings
        xl = F.relu(self.nin1l(xl))
        xl = self.nin2l(xl)
        xl = xl.flatten(2)
        xl = xl.permute(0, 2, 1)

        # right: non spatial state embeddings
        xr = xr.flatten(1)
        xr = F.relu(self.lin1r(xr))
        #xr = F.softmax(self.lin2r(xr), 0)
        xr = self.lin2r(xr)
        xr = xr.unsqueeze(-1)

        # combine left and right: weight qs
        x = xl * xr
        x = x.sum(1)

        return x
