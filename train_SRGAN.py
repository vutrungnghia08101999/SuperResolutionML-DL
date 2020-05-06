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

from models import SRCNN, ESPCN, Generator, Discriminator
from datasets import TrainDataset, EvalDataset, SRCNNTrainDataset, SRCNNEvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim
from losses import MSELoss, get_features

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
parser.add_argument('--generator-weights', type=str, required=True)
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
OUTPUT = os.path.join(args.outputs_dir, 'SRGAN', f'x{args.scale}')
os.makedirs(OUTPUT, exist_ok=True)
logging.info(f'OUTPUT: {OUTPUT}')

# *************** train and evaluate ******************
def train(train_dataset,
          train_dataloader,
          generator,
          discriminator,
          device: torch.device) -> None:
    generator.train()
    discriminator.train()
    epoch_D_losses = AverageMeter()
    epoch_G_losses = AverageMeter()
    for data in tqdm(train_dataloader):
        lr, hr = data  # batch_size x 1 x 17 x 17 and batch_size x 1 x 34 x 34, in range (0, 1) for x2

        lr = lr.to(device)
        hr = hr.to(device)

        # train discrimintor
        sr = generator(lr)  # batch_size x 1 x 34 x 34

        sr_probs_target = torch.rand(sr.shape[0], 1)*0.3 + 1e-4
        hr_probs_target = torch.rand(sr.shape[0], 1)*0.3 + 0.7 - 1e-4
        
        
        discriminator_loss = adversarial_loss(discriminator(sr), sr_probs_target) + adversarial_loss(discriminator(hr), hr_probs_target)
        epoch_D_losses.update(discriminator_loss.item(), lr.shape[0])

        optim_discriminator.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        optim_discriminator.step()

        # train generator
        sr = generator(lr)

        sr_features = get_features(vgg=vgg19,imgs=sr, vgg_depth=11)
        hr_features = get_features(vgg=vgg19,imgs=hr, vgg_depth=11)
        l1 = content_loss(sr, hr) + 0.006 * content_loss(sr_features, hr_features)

        ones = torch.ones(sr.shape[0]).view(sr.shape[0], 1) * 1.0
        l2 = adversarial_loss(discriminator(sr), ones)
        generator_loss = l1 + 0.001 * l2
        epoch_G_losses.update(generator_loss.item(), lr.shape[0])

        optim_generator.zero_grad()
        generator_loss.backward()
        optim_generator.step()

    logging.info(f'discriminator_loss: {epoch_D_losses.avg} - generator_loss: {epoch_G_losses.avg}')
    logging.info(f'Average sr: {sr_probs.mean().item()}')
    logging.info(f'Average hr: {hr_probs.mean().item()}')

    torch.save({'state_dict': discriminator.state_dict()}, os.path.join(OUTPUT, f'epoch_D_{epoch}.pth'))
    torch.save({'state_dict': generator.state_dict()}, os.path.join(OUTPUT, f'epoch_G_{epoch}.pth'))


def eval(eval_dataloader, generator, device: torch.device) -> None:
    generator.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()

    for data in eval_dataloader:
        lr, hr = data

        lr = lr.to(device)
        hr = hr.to(device)

        with torch.no_grad():
            sr = generator(lr).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(sr, hr), len(lr))
        epoch_ssim.update(calc_ssim(sr.cpu(), hr.cpu()), len(lr))

    logging.info(f'\33[91meval psnr: {epoch_psnr.avg} - eval ssim: {epoch_ssim.avg}\33[0m')


# ************* Initilize generator and discriminator, create optimizer and loss ************
generator = Generator(args.scale)
discriminator = Discriminator()
optim_generator = optim.Adam(generator.parameters(),
                             lr=args.lr,
                             betas=(0.99, 0.999))

optim_discriminator = optim.Adam(discriminator.parameters(),
                                 lr=args.lr,
                                 betas=(0.99, 0.999))
content_loss = MSELoss
adversarial_loss = nn.BCELoss()
vgg19 = torchvision.models.vgg19(pretrained=True)
for i in range(len(vgg19.features)):
    vgg19.features[i].requires_grad_(False)

# ************ create dataset and dataloader *************
train_dataset = TrainDataset(args.train_file)
eval_dataset = EvalDataset(args.eval_file)
logging.info(f'No.train patches: {len(train_dataset)}')
logging.info(f'No.eval images: {len(eval_dataset)}')

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

for epoch in range(args.num_epochs):
    train(train_dataset=train_dataset,
          train_dataloader=train_dataloader,
          generator=generator,
          discriminator=discriminator,
          device=device)
    eval(eval_dataloader=eval_dataloader, generator=generator, device=device)    

logging.info('Completed\n\n')
