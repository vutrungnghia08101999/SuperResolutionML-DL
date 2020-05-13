import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', type=int, default=4)
parser.add_argument('--test_mode', type=str, default='CPU', choices=['GPU', 'CPU'])
parser.add_argument('--image_name', type=str, required=True)
parser.add_argument('--model_path', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/models/checkpoint_20-05-09_21-25-46_50.pth')
parser.add_argument('--output', type=str, default='./')
args = parser.parse_args()

model = Generator()
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['state_dict'])

image = Image.open(args.image_name)
image = ToTensor()(image).unsqueeze(0)

with torch.no_grad():
    sr = model(image)
out_img = ToPILImage()(sr.squeeze().cpu())
out_img.save('out_srf_' + str(args.upscale_factor) + '_' + args.image_name)
