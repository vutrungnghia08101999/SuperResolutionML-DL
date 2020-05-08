import numpy as np
from PIL import Image

s = Image.open('/media/vutrungnghia/New Volume/MachineLearningAndDataMining/SuperResolution/dataset/valid-toy/2012_003367.jpg')
a = s.copy()
a = np.array(s)