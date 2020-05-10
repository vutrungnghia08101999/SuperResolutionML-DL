import os
from os import listdir
from os.path import join
from tqdm import tqdm
import math
import pickle

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromCompressFile(Dataset):
    def __init__(self, compress_file_path, crop_size, upscale_factor):
        super(TrainDatasetFromCompressFile, self).__init__()
        with open(compress_file_path, 'rb') as f:
            self.images_storage = pickle.load(f)
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.fromarray(self.images_storage[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.images_storage)

class ValDatasetFromCompressFile(Dataset):
    def __init__(self, compress_file_path: str):
        super(ValDatasetFromCompressFile, self).__init__()
        with open(compress_file_path, 'rb') as f:
            self.images_storage = pickle.load(f)

    def __getitem__(self, index):
        lr_image, hr_restore_img, hr_image = self.images_storage[index]
        return lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.images_storage)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor=4):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = os.path.join(dataset_dir, 'LR')
        self.hr_path = os.path.join(dataset_dir, 'HR')
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
