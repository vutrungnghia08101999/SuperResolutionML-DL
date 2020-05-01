import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calc_psnr(img1, img2):
    # print(type(img1))
    # print(img1.shape)
    # print(img1)
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ssim(img1, img2):
    img1 = np.array(img1.squeeze())
    img2 = np.array(img2.squeeze())
    return ssim(im1=img1, im2=img2, data_range=1)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
