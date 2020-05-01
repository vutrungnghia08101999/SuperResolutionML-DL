import argparse
import logging
import copy
import os

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
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
parser.add_argument('--train-file', type=str, required=True)
parser.add_argument('--eval-file', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--weights-file', type=str)
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

logging.info(args._get_kwargs())

if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

model = SRCNN(scale_factor=args.scale).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       betas=(0.99, 0.999))

train_dataset = TrainDataset(args.train_file)
logging.info(f'No.train patches: {len(train_dataset)}')
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

eval_dataset = EvalDataset(args.eval_file)
logging.info(f'No.valid images: {len(eval_dataset)}')
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

for epoch in range(args.num_epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data  # batch_size x 1 x 51 x 51 and batch_size x 1 x 51 x 51, in range (0, 1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save({'state_dict': model.state_dict()}, os.path.join(args.outputs_dir, f'epoch_{epoch}.pth'))

    model.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

    logging.info(f'\33[91meval psnr: {epoch_psnr.avg} - eval ssim: {epoch_ssim.avg}\33[0m')
