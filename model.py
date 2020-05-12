import math
import torch
from torch import nn


class SRResNet(nn.Module):
    def __init__(self, scale_factor=4, kernel_size=9, n_channels=64):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(SRResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels)
        )
        block4 = [UpsampleBLock(n_channels, 2) for _ in range(upsample_block_num)]
        block4.append(nn.Conv2d(n_channels, 3, kernel_size=9, padding=4))
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.residual_blocks(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block1 + block3)

        return (torch.tanh(block4) + 1) / 2

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
