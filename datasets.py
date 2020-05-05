import logging
import h5py
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

# ****************** SRCNN ********************
class SRCNNTrainDataset(Dataset):
    def __init__(self, h5_file, scale):
        super(SRCNNTrainDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = scale

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr, hr = np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)
            # return lr, hr
            s = Image.fromarray(lr.squeeze())
            Y = s.resize((s.width * self.scale, s.height * self.scale), resample=Image.BICUBIC)
            return np.expand_dims(Y, 0), hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class SRCNNEvalDataset(Dataset):
    def __init__(self, h5_file, scale=3):
        super(SRCNNEvalDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = scale

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr, hr = np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)
            s = Image.fromarray(lr.squeeze())
            Y = s.resize((s.width * self.scale, s.height * self.scale), resample=Image.BICUBIC)
            return np.expand_dims(Y, 0), hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# *************** ESPCN, SRResNet, SRGAN ****************
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
