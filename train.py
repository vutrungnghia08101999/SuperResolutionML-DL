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
import torchvision

from models import SRCNN, ESPCN, Generator
from datasets import TrainDataset, EvalDataset, SRCNNTrainDataset, SRCNNEvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim
from losses import MSELoss, perceptual_loss

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info('\n\n=========== TRAIN AND EVALUATE ============\n\n')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--loss', type=str, required=True)

parser.add_argument('--train-dir', type=str, required=True)
parser.add_argument('--eval-dir', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()
args.train_file = args.train_dir + f'_x{args.scale}.h5'
args.eval_file = args.eval_dir + f'_x{args.scale}.h5'

for k, v in args._get_kwargs():
    logging.info(f'{k}: {v}')

# ************* Set up environment ***************
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
OUTPUT = os.path.join(args.outputs_dir, args.model + '_' + args.loss, f'x{args.scale}')
# if os.path.exists(OUTPUT):
#     raise RuntimeError(f'{OUTPUT} already exist!')
os.makedirs(OUTPUT, exist_ok=True)
logging.info(f'OUTPUT: {OUTPUT}')
# ************* Choose dataset, model and loss function ************
# choose dataset
if args.model == 'SRCNN':
    train_dataset = SRCNNTrainDataset(args.train_file, args.scale)
    eval_dataset = SRCNNEvalDataset(args.eval_file, args.scale)
else:
    train_dataset = TrainDataset(args.train_file)
    eval_dataset = EvalDataset(args.eval_file)

# choose model and loss function
if args.model == 'SRCNN' and args.loss == 'mse':
    model = SRCNN(args.scale).to(device)
    criterion = MSELoss
elif args.model == 'ESPCN' and args.loss == 'mse':
    model = ESPCN(args.scale).to(device)
    criterion = MSELoss
elif args.model == 'ESPCN' and args.loss == 'vgg16_8':
    model = ESPCN(args.scale).to(device)
    criterion = perceptual_loss
    vgg16 = torchvision.models.vgg16(pretrained=True)
    for i in range(len(vgg16.features)):
        # print(len(vgg16.features))
        vgg16.features[i].requires_grad_(False)
    vgg16 = vgg16.to(device=device)
elif args.model == 'SRResNet' and args.loss == 'mse':
    model = Generator(args.scale).to(device)
    criterion = MSELoss
else:
    raise RuntimeError(f'{args.model} and {args.loss} are invalid')

# ************ create optimizer and dataloader *************
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       betas=(0.99, 0.999))

logging.info(f'No.train patches: {len(train_dataset)}')
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
logging.info(f'No.eval images: {len(eval_dataset)}')
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

# *************** train and evaluate ******************
def train(train_dataset, train_dataloader, model, loss_type: str) -> None:
    model.train()
    epoch_losses = AverageMeter()
    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data  # batch_size x 1 x 51 x 51 and batch_size x 1 x 51 x 51, in range (0, 1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            # print(preds.shape, labels.shape)
            if loss_type == 'mse':
                loss = criterion(preds, labels)
            elif loss_type == 'vgg16_8':
                # print(vgg16)
                loss = criterion(vgg=vgg16, preds=preds, labels=labels)
            else:
                raise RuntimeError(f'{loss_type} is invalid')
            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save({'state_dict': model.state_dict()}, os.path.join(OUTPUT, f'epoch_{epoch}.pth'))
    with torch.no_grad():
        logging.info(f'Loss: {epoch_losses.avg}')


def eval(eval_dataloader, model) -> None:
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


for epoch in range(args.num_epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

    train(train_dataset=train_dataset,
          train_dataloader=train_dataloader,
          model=model,
          loss_type=args.loss)
    eval(eval_dataloader=eval_dataloader, model=model)    

logging.info('Completed\n\n')
