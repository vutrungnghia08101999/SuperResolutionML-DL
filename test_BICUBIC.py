import argparse
import logging
import os
from PIL import Image

import numpy as np
import torch
import torchvision.utils as TV_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor

from dataset import TestDatasetFromFolder, display_transform
# from model import Generator
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel, AverageMeter

logging.basicConfig(filename='logs/test_BICUBIC.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H-%M-%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--test_folder', type=str, default='dataset/test')
parser.add_argument('--test_dataset', type=str, default='BSD100,Manga109,Set5,Set14,Urban100')
args = parser.parse_args()

logging.info('\n\n================ BICUBIC - TESTING =================\n\n')
logging.info(args._get_kwargs())

images_output = f'experiments/BICUBIC/test'
os.makedirs(images_output, exist_ok=True)
logging.info(f'OUTPUT: {images_output}')

psnrs = AverageMeter()
ssims = AverageMeter()
datasets = args.test_dataset.split(',')
for dataset in datasets:
    logging.info(f'Test on: {dataset}')
    output = os.path.join(images_output, dataset)
    os.makedirs(output, exist_ok=True)
    logging.info(f'output: {output}')
    test_folder = f'dataset/test/{dataset}/SRF_4'
    test_set = TestDatasetFromFolder(test_folder)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    for image_name, lr, hr_bicubic, hr in tqdm(test_loader):
        image_name = image_name[0]
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        lr = ToPILImage()(lr.squeeze())
        W, H = lr.size
        sr = lr.resize((W*4, H*4), resample=Image.BICUBIC)
        sr = ToTensor()(sr)
        
        sr_img = ToPILImage()(sr.cpu().squeeze())
        hr_img = ToPILImage()(hr.cpu().squeeze())
        psnr = calculate_psnr_y_channel(sr_img, hr_img)
        ssim = calculate_ssim_y_channel(sr_img, hr_img)
        psnrs.update(psnr)
        ssims.update(ssim)

        test_images = torch.stack([display_transform()(hr_bicubic.squeeze(0)),
                                display_transform()(hr.cpu().squeeze(0)),
                                display_transform()(sr.cpu().squeeze(0))])
        image = TV_utils.make_grid(test_images, nrow=3, padding=5)
        path = os.path.join(output, image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1])
        TV_utils.save_image(image, path, padding=5)
    logging.info(f'PSNR: {psnrs.avg} - SSIM: {ssims.avg}')
logging.info('Completed\n\n')