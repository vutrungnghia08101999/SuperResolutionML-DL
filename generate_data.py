import logging
import pickle
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from dataset import calculate_valid_crop_size, train_hr_transform, train_lr_transform, display_transform

logging.basicConfig(filename='logs/generate_data.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ************************* GENERATE TEST DATASET ***************************
logging.info("GENERATE TEST DATA")
UPSCALE_FACTOR = 4
ROOT = 'dataset/test'
DATASETS = ['Manga109', 'BSD100', 'Set5', 'Set14', 'Urban100']

logging.info(f'UPSCALE_FACTOR: {UPSCALE_FACTOR}')
logging.info(f'ROOT: {ROOT}')
logging.info(f'DATASETS: {DATASETS}')

for dataset in DATASETS:
    logging.info(f'Processing {dataset}')
    dataset_folder = os.path.join(ROOT, dataset)
    input_folder = os.path.join(dataset_folder, 'original')
    output_folder = os.path.join(dataset_folder, f'SRF_{UPSCALE_FACTOR}')
    lr_folder = os.path.join(output_folder, 'LR')
    hr_folder = os.path.join(output_folder, 'HR')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)
    os.makedirs(hr_folder, exist_ok=True)
    images = list(os.listdir(input_folder))
    logging.info(f'No.images: {len(images)}')
    logging.info(f'INPUT: {input_folder}')
    logging.info(f'OUTPUT: {output_folder}')
    for image_name in tqdm(images):
        img = Image.open(os.path.join(input_folder, image_name))
        width, height = img.size
        crop_size = min(width, height) - min(width, height) % UPSCALE_FACTOR
        hr = CenterCrop(crop_size)(img)
        lr = Resize(crop_size // UPSCALE_FACTOR, interpolation=Image.BICUBIC)(hr)
        lr.save(os.path.join(lr_folder, image_name))
        hr.save(os.path.join(hr_folder, image_name))
logging.info('Completed generate test data')

# ************************* GENERATE TRAIN DATASET ****************************
logging.info('GENERATE TRAIN DATA')
ROOT = 'dataset/train'
TRAIN = 'VOC-2012-train'
VALID = 'VOC-2012-valid'
UPSCALE_FACTOR = 4

logging.info(f'ROOT: {ROOT}')
logging.info(f'TRAIN: {TRAIN}')
logging.info(f'VALID: {VALID}')
logging.info(f'UPSCALE_FACTOR: {UPSCALE_FACTOR}')

# ************** COMPRESS TRAIN ******************

def load_data(dataset_dir: str):
    images_storage = []
    files = os.listdir(dataset_dir)
    logging.info(f'Loading {len(files)} images from {dataset_dir}')
    for filename in tqdm(files):
        img = Image.open(os.path.join(dataset_dir, filename))
        images_storage.append(np.array(img))
        img.close()
    return images_storage

images_storage = load_data(dataset_dir=os.path.join(ROOT, TRAIN))
compress_path = os.path.join(ROOT, f'{TRAIN}.pkl')
with open(compress_path, 'wb') as f:
    pickle.dump(images_storage, f)
logging.info(compress_path)

# ************** COMPRESS VALID ****************
valid_storages = []
files = list(os.listdir(os.path.join(ROOT, VALID)))
for filename in tqdm(files):
    hr_image = Image.open(os.path.join(ROOT, VALID, filename))
    w, h = hr_image.size
    crop_size = calculate_valid_crop_size(min(w, h), UPSCALE_FACTOR)
    lr_scale = Resize(crop_size // UPSCALE_FACTOR, interpolation=Image.BICUBIC)
    hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
    hr_image = CenterCrop(crop_size)(hr_image)
    lr_image = lr_scale(hr_image)
    hr_restore_img = hr_scale(lr_image)
    valid_storages.append(
        (ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image))
    )

compress_path = os.path.join(ROOT, f'{VALID}_{UPSCALE_FACTOR}.pkl')
with open(compress_path, 'wb') as f:
    pickle.dump(valid_storages, f)
logging.info(compress_path)
logging.info('DONE GENERATE DATA')
