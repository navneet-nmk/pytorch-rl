# Training Script for the InfoGAN

import torch
# Add this line to get better performance
torch.backends.cudnn.benchmark=True
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from models import infogan

USE_CUDA = torch.cuda.is_available()


class MontezumaRevengeFramesDataset(Dataset):
    """

    Dataset consisting of the frames of the Atari Game-
    Montezuma Revenge

    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.list_files()

    def __len__(self):
        return len(self.images)

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
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}


if __name__ == '__main__':
    dataset = MontezumaRevengeFramesDataset(root_dir='/mr', transform=transforms.Compose([Rescale(256), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    model = infogan.InfoGAN(conv_layers=32,
                              conv_kernel_size=3,
                              generator_input_channels=1,
                              generator_output_channels=3,
                              batch_size=4, categorical_dim=10, continuous_dim=2,
                              pool_kernel_size=3, height=256, width=256, discriminator_input_channels=3,
                              discriminator_lr=1e-4, generator_lr=1e-4, discriminator_output_dim=1,
                              output_dim=12, hidden_dim=256, num_epochs=100)

    model.train(dataloader=dataloader)

