import argparse
import logging
import copy
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import ESPCN
from datasets import TrainDataset, EvalDataset
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
parser.add_argument('--weights-file', type=str)
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

logging.info(args._get_kwargs())

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

model = ESPCN(scale_factor=args.scale).to(device)
checkpoint = torch.load(args.weights_file, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

test_dataset = EvalDataset(args.test_file)
logging.info(f'No.test images: {len(test_dataset)}')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

model.eval()
epoch_psnr = AverageMeter()
epoch_ssim = AverageMeter()

for data in tqdm(test_dataloader):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = model(inputs).clamp(0.0, 1.0)

    epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
    epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

logging.info(f'\33[91meval psnr: {epoch_psnr.avg} - eval ssim: {epoch_ssim.avg}\33[0m')
