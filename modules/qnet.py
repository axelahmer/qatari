import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class QNet(nn.Module):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer
        self.logging = False

    def log_forward(self, x):
        self.logging = True
        y = self.forward(x)
        self.logging = False
        return y
