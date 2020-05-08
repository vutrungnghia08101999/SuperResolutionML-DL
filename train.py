import argparse
import os
from math import log10
import time

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TrainDatasetFromCompressFile, ValDatasetFromCompressFile
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
# parser.add_argument('--train_folder', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/VOC-2012-train')
# parser.add_argument('--valid_folder', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/VOC-2012-valid')
parser.add_argument('--train_file', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/train-toy_4_88.pkl')
parser.add_argument('--valid_file', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid-toy_4.pkl')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/experiments/SRGAN')
opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

os.makedirs(opt.output, exist_ok=True)

# train_set = TrainDatasetFromFolder(opt.train_folder, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
# val_set = ValDatasetFromFolder(opt.valid_folder, upscale_factor=UPSCALE_FACTOR)
# train_set = TrainDatasetFromFolder(opt.train_folder, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
# val_set = ValDatasetFromFolder(opt.valid_folder, upscale_factor=UPSCALE_FACTOR)

train_set = TrainDatasetFromCompressFile(opt.train_file)
val_set = ValDatasetFromCompressFile(opt.valid_file)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    s = time.time() * 1000
    for data, target in train_bar:
        e = time.time() * 1000
        # print(f'Load data: {e - s}')
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        
        optimizerG.step()

        s = time.time() * 1000
        # print(f'Training time: {s - e}')
        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    netG.eval()
    # save images
    out_imgs_path = os.path.join(opt.output, f'training_results/SRF_{UPSCALE_FACTOR}')
    if not os.path.exists(out_imgs_path):
        os.makedirs(out_imgs_path)
    
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        counter = 0
        for val_lr, val_hr_restore, val_hr in val_bar:
            counter += 1
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)
    
            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))
            if counter % 100 == 0:
                val_images.extend([
                    display_transform()(val_hr_restore.squeeze(0)),
                    display_transform()(hr.data.cpu().squeeze(0)),
                    display_transform()(sr.data.cpu().squeeze(0))])
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, os.path.join(out_imgs_path, f'epoch_{epoch}_{index}.png'), padding=5)
            index += 1

    # save model parameters
    models_path = os.path.join(opt.output, 'models')
    os.makedirs(models_path, exist_ok=True)
    torch.save(netG.state_dict(), os.path.join(models_path, f'netG_epoch_{UPSCALE_FACTOR}_{epoch}.pth'))
    torch.save(netD.state_dict(), os.path.join(models_path, f'netD_epoch_{UPSCALE_FACTOR}_{epoch}.pth'))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    statistics_path = os.path.join(opt.output, 'statistics')
    os.makedirs(statistics_path, exist_ok=True)
    if epoch % 10 == 0 and epoch != 0:
        data_frame = pd.DataFrame(
            data={
                'Loss_D': results['d_loss'],
                'Loss_G': results['g_loss'],
                'Score_D': results['d_score'],
                'Score_G': results['g_score'],
                'PSNR': results['psnr'],
                'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(statistics_path, f'srf_{UPSCALE_FACTOR}_train_results.csv'), index='Epoch')
