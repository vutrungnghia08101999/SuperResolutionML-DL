import math
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, kernel_size=3):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Tanh(),
            # nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Tanh(),
            # nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Tanh(),
            # nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x
