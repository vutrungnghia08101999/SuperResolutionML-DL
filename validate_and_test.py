import argparse
import logging
import copy
import os
from tqdm import tqdm
import PIL.Image as pil_image

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--test-file', type=str, required=True)
parser.add_argument('--valid-file', type=str, required=True)
parser.add_argument('--weights-file', type=str)
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

logging.info(args._get_kwargs())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

valid_dataset = EvalDataset(args.valid_file)
logging.info(f'No.valid images: {len(valid_dataset)}')
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)

test_dataset = EvalDataset(args.test_file)
logging.info(f'No.test images: {len(test_dataset)}')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

epoch_psnr = AverageMeter()
epoch_ssim = AverageMeter()

for data in tqdm(valid_dataloader):
    inputs, labels = data
    inputs = inputs.to(device).squeeze()
    inputs = inputs.clamp(0.0, 1.0)
    inputs = pil_image.fromarray(inputs.numpy())

    labels = labels.to(device).squeeze()
    labels = labels.clamp(0.0, 1.0)
    labels = labels.numpy()

    bicubic = inputs.resize((inputs.width * args.scale, inputs.height * args.scale), resample=pil_image.BICUBIC)
    bicubic = np.array(bicubic)
    epoch_psnr.update(calc_psnr(torch.tensor(bicubic), torch.tensor(labels)), 1)
    epoch_ssim.update(calc_ssim(bicubic, labels), 1)

logging.info(f'\33[91mvalid psnr: {epoch_psnr.avg} - valid ssim: {epoch_ssim.avg}\33[0m')

# *******************************************************8
epoch_psnr = AverageMeter()
epoch_ssim = AverageMeter()

for data in tqdm(test_dataloader):
    inputs, labels = data
    inputs = inputs.to(device).squeeze()
    inputs = inputs.clamp(0.0, 1.0)
    inputs = pil_image.fromarray(inputs.numpy())

    labels = labels.to(device).squeeze()
    labels = labels.clamp(0.0, 1.0)
    labels = labels.numpy()

    bicubic = inputs.resize((inputs.width * args.scale, inputs.height * args.scale), resample=pil_image.BICUBIC)
    bicubic = np.array(bicubic)
    epoch_psnr.update(calc_psnr(torch.tensor(bicubic), torch.tensor(labels)), 1)
    epoch_ssim.update(calc_ssim(bicubic, labels), 1)

logging.info(f'\33[91mtest psnr: {epoch_psnr.avg} - test ssim: {epoch_ssim.avg}\33[0m')
