import pickle
from PIL import Image
import os
from tqdm import tqdm

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

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

ROOT = '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset'
crop_size = 88
upscale_factor = 4

# ************** COMPRESS TRAIN ******************

def load_data(dataset_dir: str, hr_transform, lr_transform):
    images_storage = []
    files = os.listdir(dataset_dir)
    print(f'Loading {len(files)} images from {dataset_dir}')
    for filename in tqdm(files):
        try:
            img = Image.open(os.path.join(dataset_dir, filename))
            hr_image = hr_transform(img)
            lr_image = lr_transform(hr_image)
            images_storage.append((lr_image, hr_image))
            img.close()
        except Exception as e:
            print(filename)
    return images_storage

crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
hr_transform = train_hr_transform(crop_size)
lr_transform = train_lr_transform(crop_size, upscale_factor)

for TRAIN in ['VOC-2012-train', 'train-toy']:
    images_storage = load_data(dataset_dir=os.path.join(ROOT, TRAIN),
                            hr_transform=hr_transform,
                            lr_transform=lr_transform)
    COMPRESS_PATH = os.path.join(ROOT, f'{TRAIN}_{upscale_factor}_{crop_size}.pkl')
    with open(COMPRESS_PATH, 'wb') as f:
        pickle.dump(images_storage, f)

# ************** COMPRESS VALID ****************

for VALID in ['VOC-2012-valid', 'valid-toy']:
    valid_storages = []
    files = list(os.listdir(os.path.join(ROOT, VALID)))
    for filename in tqdm(files):
        hr_image = Image.open(os.path.join(ROOT, VALID, filename))
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), upscale_factor)
        lr_scale = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        valid_storages.append(
            (ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image))
        )

    COMPRESS_PATH = os.path.join(ROOT, f'{VALID}_{upscale_factor}.pkl')
    with open(COMPRESS_PATH, 'wb') as f:
        pickle.dump(valid_storages, f)
