import argparse
import logging
import os

import numpy as np
import torch
import torchvision.utils as TV_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from dataset import TestDatasetFromFolder, display_transform
from model import ESPCN
from utils import calculate_psnr_y_channel, calculate_ssim_y_channel, AverageMeter

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H-%M-%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/models/checkpoint_20-05-17_07-50-00_100.pth')
parser.add_argument('--test_folder', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/BSDS100/SRF_4')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/testing-world-champion/ESPCN/X4')
args = parser.parse_args()

logging.info('\n\n================ ESPCN - TESTING =================\n\n')
logging.info(args._get_kwargs())

model = ESPCN()
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['state_dict'])
logging.info(model)

if torch.cuda.is_available():
    model = model.cuda()

test_set = TestDatasetFromFolder(args.test_folder)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

model.eval()
psnrs = AverageMeter()
ssims = AverageMeter()
for image_name, lr, hr_bicubic, hr in tqdm(test_loader):
    image_name = image_name[0]
    if torch.cuda.is_available():
        lr = lr.cuda()
        hr = hr.cuda()
    with torch.no_grad():
        sr = model(lr)
        sr = torch.clamp(sr, 0.0, 1.0)
    
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
    path = os.path.join(args.output, image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1])
    TV_utils.save_image(image, path, padding=5)
logging.info(f'PSNR: {psnrs.avg} - SSIM: {ssims.avg}')
logging.info('Completed\n\n')
