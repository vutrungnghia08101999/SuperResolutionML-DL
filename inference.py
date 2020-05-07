import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name', required=True)
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/experiments/SRGAN')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.output, 'models', MODEL_NAME)))
else:
    model.load_state_dict(torch.load(os.path.join(opt.output, 'models', MODEL_NAME), map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = ToTensor()(image).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.clock()
out = model(image)
elapsed = (time.clock() - start)
print(f'cost {elapsed} s')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
