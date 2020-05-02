import argparse
import logging
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr, calc_ssim

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--weights-file', type=str, required=True)
parser.add_argument('--image-folder', type=str, required=True)
parser.add_argument('--scale', type=int, default=3)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ESPCN(scale_factor=args.scale).to(device)

checkpoint = torch.load(args.weights_file)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

logging.info(args._get_kwargs())

for filename in os.listdir(args.image_folder):
    logging.info(f'{filename}')
    image_file = os.path.join(args.image_folder, filename)

    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    ssim = calc_ssim(hr, preds)
    logging.info(f'pnsr: {psnr} - ssim: {ssim}')
    preds = preds.mul(255.0).cpu().numpy().squeeze()

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    path = image_file.replace('.', '_x{}.'.format(args.scale))
    output.save(path)
    logging.info(f'Save at: {path}')

logging.info('Completed')
