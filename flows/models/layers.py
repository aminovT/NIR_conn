import torch
import torch.nn
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self, dim_in):
        super(BatchNorm, self).__init__()

        self.mu = nn.Parameter(torch.zeros(dim_in), requires_grad=False)
        self.sig2 = nn.Parameter(torch.zeros(dim_in) + 0.1, requires_grad=False)

        self.momentum = 0.1

    def forward(self, x):

        if self.training:
            mu = x.mean(0)
            sig2 = (x - mu).pow(2).mean(0)
            x = (x - mu) / (sig2 + 1.0e-6).sqrt()
            self.mu = nn.Parameter(self.momentum * mu + (1 - self.momentum) * self.mu)
            self.sig2 = nn.Parameter(self.momentum * sig2 + (1 - self.momentum) * self.sig2)
            return x, sig2 + 1.0e-6
        else:
            x = (x - self.mu) / (self.sig2 + 1.0e-6).sqrt()
            return x, self.sig2 + 1.0e-6


class SNet(nn.Module):
    def __init__(self, dim_in, dim_middle):
        super(SNet, self).__init__()
        self.h = nn.LeakyReLU() #nn.Tanh()  # nn.LeakyReLU()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_middle),
            self.h,
            nn.Linear(dim_middle, dim_in)
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.clamp(x, -1, 1)

        return x


class TNet(nn.Module):
    def __init__(self, dim_in, dim_middle):
        super(TNet, self).__init__()
        self.h = nn.LeakyReLU() # nn.Tanh()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_middle),
            self.h,
            nn.Linear(dim_middle, dim_in),
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.clamp(x, -1, 1)
        return x
