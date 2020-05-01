import math
from torch import nn
import torch.nn.functional as F
from PIL import Image


class SRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(9, 9), padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(5, 5), padding=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
