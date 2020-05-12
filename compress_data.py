import pickle
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from dataset import calculate_valid_crop_size, train_hr_transform, train_lr_transform, display_transform


# ROOT = '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/hyperparameter-tuning'
ROOT = '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset'
# ************** COMPRESS TRAIN ******************

# def load_data(dataset_dir: str):
#     images_storage = []
#     files = os.listdir(dataset_dir)
#     print(f'Loading {len(files)} images from {dataset_dir}')
#     for filename in tqdm(files):
#         img = Image.open(os.path.join(dataset_dir, filename))
#         images_storage.append(np.array(img))
#         img.close()
#     return images_storage

# # for TRAIN in ['VOC-2012-train', 'train-toy']:
# for TRAIN in ['VOC-2012-train']:
#     images_storage = load_data(dataset_dir=os.path.join(ROOT, TRAIN))

#     COMPRESS_PATH = os.path.join(ROOT, f'{TRAIN}.pkl')
#     with open(COMPRESS_PATH, 'wb') as f:
#         pickle.dump(images_storage, f)

# ************** COMPRESS VALID ****************
upscale_factor = 4

# for VALID in ['VOC-2012-valid', 'valid-toy']:
for VALID in ['VOC-2012-valid']:
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
