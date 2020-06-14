import argparse
import logging
import os

import numpy as np
from PIL import Image
import torch
import torchvision.utils as TV_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor

from dataset import TestDatasetFromFolder, display_transform
from models import ESPCN, SRResNet, Generator
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel, AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--model', type=str, choices=['ESPCN', 'SRResNet', 'SRGAN'])
args = parser.parse_args()

print(args._get_kwargs())

if args.model == 'ESPCN':
    model = ESPCN()
elif args.model == 'SRResNet':
    model = SRResNet()
else:
    model = Generator()
model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu'))['state_dict'])

if torch.cuda.is_available():
    model = model.cuda()

image = Image.open(args.input)
lr = ToTensor()(image)
lr = lr.unsqueeze(0)
print('Forwarding ...')
with torch.no_grad():
    sr = model(lr)
    sr = torch.clamp(sr, 0.0, 1.0)

sr_img = ToPILImage()(sr.cpu().squeeze())
sr_img.save(f'{args.model}.png')
print('Completed\n')
