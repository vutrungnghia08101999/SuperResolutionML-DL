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
from models import ESPCN
from utils import AverageMeter, calculate_psnr_y_channel, calculate_ssim_y_channel

logging.basicConfig(filename='logs/train_ECPCN.txt',
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
parser.add_argument('--models_dir', type=str, default='models/ESPCN')
parser.add_argument('--weights', type=str, default='')
parser.add_argument('--images', type=str, default='experiments/ESPCN/train')
args = parser.parse_args()

logging.info('\n\n================ ESPCN - TRAINING =================\n\n')
logging.info(args._get_kwargs())

os.makedirs(args.models_dir,exist_ok=True)
os.makedirs(args.images, exist_ok=True)

train_set = TrainDatasetFromCompressFile(args.train_file, args.crop_size, args.upscale_factor)
val_set = ValDatasetFromCompressFile(args.valid_file)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

model = ESPCN()
logging.info(f'No.parameters: {sum(param.numel() for param in model.parameters())}')
logging.info(model)
if args.weights:
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f'Loading model at: {args.weights}')

criterion = torch.nn.MSELoss()

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters())

for epoch in range(1, args.num_epochs + 1):
    model.train()
    train_losses = AverageMeter()
    for lrs, hrs in tqdm(train_loader):
        if torch.cuda.is_available():
            lrs, hrs = lrs.cuda(), hrs.cuda()

        srs = model(lrs)

        loss = criterion(srs, hrs)
        train_losses.update(loss.item(), lrs.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info(f'EPOCH {epoch} - Loss: {train_losses.avg}')
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H-%M-%S")
    model_checkpoint = os.path.join(args.models_dir, f'checkpoint_{time}_{epoch}.pth')
    torch.save({'state_dict': model.state_dict()}, model_checkpoint)
    logging.info(model_checkpoint)

    model.eval()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    valid_losses = AverageMeter()
    for image_id, (lr, hr_bicubic, hr) in tqdm(enumerate(val_loader)):
        assert lr.shape[0] == 1
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()

        with torch.no_grad():
            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0)
        
        loss = criterion(sr, hr)
        valid_losses.update(loss)
        sr_img = ToPILImage()(sr.cpu().squeeze())
        hr_img = ToPILImage()(hr.cpu().squeeze())
        if image_id == 0:
            s = torch.cat((ToTensor()(sr_img), ToTensor()(hr_img)), dim=2)
            s = ToPILImage()(s)
            s.save(f'{args.images}/{epoch}.png')
        psnr = calculate_psnr_y_channel(sr_img, hr_img)
        ssim = calculate_ssim_y_channel(sr_img, hr_img)
        
        psnrs.update(psnr)
        ssims.update(ssim)
    logging.info(f'Val_loss: {valid_losses.avg} - PSNR: {psnrs.avg} - SSIM: {ssims.avg}')
logging.info('Completed\n\n')
