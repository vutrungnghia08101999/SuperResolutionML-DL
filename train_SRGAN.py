import argparse
import logging
import os
import time
import datetime

import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor

from dataset import display_transform, TrainDatasetFromCompressFile, ValDatasetFromCompressFile
from GeneratorLoss import GeneratorLoss
from models import Generator, Discriminator
from utils import AverageMeter, calculate_psnr_y_channel, calculate_ssim_y_channel

logging.basicConfig(filename='logs/train_SRGAN.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', type=int, default=88)
parser.add_argument('--upscale_factor', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--train_file', type=str, default='dataset/train/VOC-2012-train.pkl')
parser.add_argument('--valid_file', type=str, default='dataset/train/VOC-2012-valid_4.pkl')
parser.add_argument('--models_dir', type=str, default='models/SRGAN')
parser.add_argument('--weights_G', type=str, default='')
parser.add_argument('--weights_D', type=str, default='')
parser.add_argument('--images', type=str, default='experiments/SRGAN/train')
args = parser.parse_args()


logging.info('\n\n================ SRGAN - TRAINING =================\n\n')
logging.info(args._get_kwargs())

os.makedirs(args.models_dir,exist_ok=True)
os.makedirs(args.images, exist_ok=True)

train_set = TrainDatasetFromCompressFile(args.train_file, args.crop_size, args.upscale_factor)
val_set = ValDatasetFromCompressFile(args.valid_file)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

generator = Generator()
logging.info(f'generator parameters: {sum(param.numel() for param in generator.parameters())}')
discriminator = Discriminator()
logging.info(f'discriminator parameters: {sum(param.numel() for param in discriminator.parameters())}')
logging.info(generator)
logging.info(discriminator)

if args.weights_G:
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.weights_G, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.weights_G)
    generator.load_state_dict(checkpoint['state_dict'])
    logging.info(f'Loading model at: {args.weights_G}')

if args.weights_D:
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.weights_D, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.weights_D)
    discriminator.load_state_dict(checkpoint['state_dict'])
    logging.info(f'Loading model at: {args.weights_D}')

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(generator.parameters())
optimizerD = optim.Adam(discriminator.parameters())

for epoch in range(1, args.num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_d_loss = AverageMeter()
    avg_hr_prob = AverageMeter()
    avg_sr_prob = AverageMeter()
    avg_g_loss = AverageMeter()
    for lrs, hrs in tqdm(train_loader):
        batch_size = lrs.shape[0]

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        if torch.cuda.is_available():
            lrs = lrs.cuda()
            hrs = hrs.cuda()
        srs = generator(lrs)

        discriminator.zero_grad()
        hr_prob = discriminator(hrs).mean()
        sr_prob = discriminator(srs).mean()
        d_loss = 1 - hr_prob + sr_prob

        avg_d_loss.update(d_loss.item(), batch_size)
        avg_hr_prob.update(hr_prob.item(), batch_size)
        avg_sr_prob.update(sr_prob.item(), batch_size)

        d_loss.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        generator.zero_grad()
        srs = generator(lrs)
        sr_prob = discriminator(srs).mean()
        g_loss = generator_criterion(sr_prob, srs, hrs)
        avg_g_loss.update(g_loss, batch_size)
        
        g_loss.backward()
        
        optimizerG.step()

    logging.info(f'EPOCH: {epoch} - D_loss: {avg_d_loss.avg} - sr_prob: {avg_sr_prob.avg} - hr_prob: {avg_hr_prob.avg} - G_loss: {avg_g_loss.avg}')
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H-%M-%S")
    checkpoint_G = os.path.join(args.models_dir, f'G_checkpoint_{time}_{epoch}.pth')
    torch.save({'state_dict': generator.state_dict()}, checkpoint_G)
    logging.info(checkpoint_G)

    checkpoint_D = os.path.join(args.models_dir, f'D_checkpoint_{time}_{epoch}.pth')
    torch.save({'state_dict': discriminator.state_dict()}, checkpoint_D)
    logging.info(checkpoint_D)

    generator.eval()
    discriminator.eval()

    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()
    avg_sr_prob = AverageMeter()
    avg_hr_prob = AverageMeter()
    for image_id, (lr, hr_bicubic, hr) in tqdm(enumerate(val_loader)):
        assert lr.shape[0] == 1
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()

        with torch.no_grad():
            sr = generator(lr)
            sr_prob = discriminator(sr)
            hr_prob = discriminator(hr)
        
        sr_img = ToPILImage()(sr.cpu().squeeze())
        hr_img = ToPILImage()(hr.cpu().squeeze())
        if image_id == 0:
            s = torch.cat((ToTensor()(sr_img), ToTensor()(hr_img)), dim=2)
            s = ToPILImage()(s)
            s.save(f'{args.images}/{epoch}.png')
        psnr = calculate_psnr_y_channel(sr_img, hr_img)
        ssim = calculate_ssim_y_channel(sr_img, hr_img)
        
        avg_psnr.update(psnr)
        avg_ssim.update(ssim)
        avg_sr_prob.update(sr_prob.item())
        avg_hr_prob.update(hr_prob.item())
    logging.info(f'PSNR: {avg_psnr.avg} - SSIM: {avg_ssim.avg} - SR_PROB: {avg_sr_prob.avg} - HR_PROB: {avg_hr_prob.avg}')
logging.info('Completed\n\n')
