import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale-factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_10.pth', type=str, help='generator model epoch name')
parser.add_argument('--test-folder', default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/BSDS100')
parser.add_argument('--output', type=str, default='/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/experiments/SRGAN')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(os.path.join(opt.output, 'models', MODEL_NAME)))

test_set = TestDatasetFromFolder(opt.test_folder, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

imgs_path = os.path.join(opt.output, f'benchmark_results/SRF_{UPSCALE_FACTOR}')
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

    test_images = torch.stack([display_transform()(hr_restore_img.squeeze(0)),
                               display_transform()(hr_image.data.cpu().squeeze(0)),
                               display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    path = os.path.join(imgs_path, image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1])
    utils.save_image(image, path, padding=5)

    # save psnr\ssim
    results['BSD100']['psnr'].append(psnr)
    results['BSD100']['ssim'].append(ssim)

statistics_path = os.path.join(opt.output, 'statistics')
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(os.path.join(statistics_path, f'srf_{UPSCALE_FACTOR}_test_results.csv'), index_label='DataSet')
