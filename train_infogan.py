# Training Script for the InfoGAN

import torch
# Add this line to get better performance
torch.backends.cudnn.benchmark=True
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os

USE_CUDA = torch.cuda.is_available()


class MontezumaRevengeFramesDataset(Dataset):
    """

    Dataset consisting of the frames of the Atari Game-
    Montezuma Revenge

    """

    def __init__(self, root_dir, length, transform=None):
        self.length = length
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.list_files()

    def __len__(self):
        return self.length

    def list_files(self):
        for m in os.listdir(self.root_dir):
            if m.endswith('.jpg'):
                self.images.append(m)

    def __getitem__(self, idx):
        m = self.images[idx]
        image = io.imread(os.path.join( self.root_dir, m))
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Transformations




if __name__ == '__main__':
    pass

