import math
import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity
from torchvision.transforms import ToTensor

def convert_to_y_channel(img: Image.Image):
    """Conver and RGB image to y channel, the output in range (0, 255)
    """
    img = np.array(img.convert('RGB'))
    return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256


def calculate_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """
    Arguments:
        img1 {Image.Image} -- RGB image
        img2 {Ima
    """
    img1 = ToTensor()(img1.convert('RGB'))
    img2 = ToTensor()(img2.convert('RGB'))
    return 10 * math.log10(1.0/((img1 - img2)**2).mean() + 1e-8)  # makesure img1 and img2 in range(0, 1)


def calculate_ssim_y_channel(img1: Image.Image, img2: Image.Image):
    y_img1 = convert_to_y_channel(img1)
    y_img2 = convert_to_y_channel(img2)
    return structural_similarity(y_img1 / 255.0, y_img2 / 255.0)  # makesure y_img1 and y_img2 in range(0, 1)


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
