from models.blocks import ConvReLUBatchNorm
from models.text import EfficientPunctBERT
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientPunct(nn.Module):

    def __init__(self):
        super(EfficientPunct, self).__init__()
        self.eptdnn = EfficientPunctTDNN()
        self.epbert = EfficientPunctBERT()
        self.alpha = 0.4

    def forward(self, x):
        x_tdnn = self.eptdnn(x)
        x_bert = self.epbert(x[:, :768, 150])
        p_tdnn = F.softmax(x_tdnn, dim=1)
        p_bert = F.softmax(x_bert, dim=1)
        p = self.alpha * p_tdnn + (1 - self.alpha) * p_bert
        return p

class EfficientPunctTDNN(nn.Module):

    def __init__(self):
        super(EfficientPunctTDNN, self).__init__()
        self.linear0 = nn.Linear(1792, 512)
        self.tdnn = nn.Sequential(
            ConvReLUBatchNorm(in_channels=512, out_channels=256, kernel_size=9),
            ConvReLUBatchNorm(in_channels=256, out_channels=256, kernel_size=9, dilation=2),
            ConvReLUBatchNorm(in_channels=256, out_channels=128, kernel_size=5),
            ConvReLUBatchNorm(in_channels=128, out_channels=128, kernel_size=5, dilation=2),
            ConvReLUBatchNorm(in_channels=128, out_channels=64, kernel_size=7),
            ConvReLUBatchNorm(in_channels=64, out_channels=64, kernel_size=7, dilation=2),
            ConvReLUBatchNorm(in_channels=64, out_channels=4, kernel_size=5)
        )
        self.linear1 = nn.Linear(243, 70)
        self.batchnorm = nn.BatchNorm1d(70)
        self.linear2 = nn.Linear(70, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear0(x)
        x = F.relu(x)
        x = torch.transpose(x, 1, 2)

        x = self.tdnn(x)

        x0 = self.linear1(x[:, 0, :])
        x0 = F.relu(x0)
        x0 = self.batchnorm(x0)

        x1 = self.linear1(x[:, 1, :])
        x1 = F.relu(x1)
        x1 = self.batchnorm(x1)

        x2 = self.linear1(x[:, 2, :])
        x2 = F.relu(x2)
        x2 = self.batchnorm(x2)

        x3 = self.linear1(x[:, 3, :])
        x3 = F.relu(x3)
        x3 = self.batchnorm(x3)

        x0 = self.linear2(x0)
        x1 = self.linear2(x1)
        x2 = self.linear2(x2)
        x3 = self.linear2(x3)

        x = torch.hstack((x0, x1, x2, x3))
        return x
