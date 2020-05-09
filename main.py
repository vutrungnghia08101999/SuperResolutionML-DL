from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
PATH = '/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/hyperparameter-tuning/valid-toy/2011_004766.jpg'

img = Image.open(PATH)

f = display_transform()

s = f(ToTensor()(img))




s = convert_to_y_channel(a)