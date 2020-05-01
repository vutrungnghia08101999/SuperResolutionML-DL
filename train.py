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
args = parser.parse_args([
    '--train-file', '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train/t91.h5',
    '--eval-file', '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid/Set14.h5',
    '--outputs-dir', '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/outputs'
])

args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

logging.info(args._get_kwargs())

if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

model = ESPCN(scale_factor=args.scale).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.first_part.parameters()},
    {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
], lr=args.lr)

train_dataset = TrainDataset(args.train_file)
logging.info(f'No.patches: {len(train_dataset)}')
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
eval_dataset = EvalDataset(args.eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

best_weights = copy.deepcopy(model.state_dict())
# best_epoch = 0
# best_psnr = 0.0

for epoch in range(args.num_epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

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

    torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

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

    # if epoch_psnr.avg > best_psnr:
    #     best_epoch = epoch
    #     best_psnr = epoch_psnr.avg
    #     best_weights = copy.deepcopy(model.state_dict())

# logging.info('best epoch: {}, psnr: {:.4f}'.format(best_epoch, best_psnr))
# torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
