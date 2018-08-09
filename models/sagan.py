"""

This file contains the implementation of the Self attention GAN

https://arxiv.org/abs/1805.08318

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


class SAGAN(object):

    def __init__(self, generator, discriminator,
                 dataset, num_epochs,
                 random_seed, shuffle, use_cuda,
                 tensorboard_summary_writer,
                 output_folder,
                 generator_lr, discriminator_lr, batch_size):


        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.seed = random_seed
        self.batch = batch_size
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.generator = generator
        self.discriminator = discriminator
        self.tb_writer = tensorboard_summary_writer
        self.output_folder = output_folder

        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        self.gen_optim = Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=generator_lr)
        self.dis_optim = Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=discriminator_lr)

    def set_seed(self):
        # Set the seed for reproducible results
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def get_dataloader(self):
        # Generates the dataloader for the images for training

        dataset_loader = DataLoader(self.dataset,
                                    batch_size=self.batch,
                                    shuffle=self.shuffle)

        return dataset_loader


    # Discriminator Loss function
    def disc_loss(self):

        criterionD = nn.BCELoss()
        return criterionD

    # Noise sample generator
    def _noise_sample(self, noise, bs):
        noise.data.uniform_(-1.0, 1.0)
        z = noise

        return z

    def linear_annealing_variance(self, std, epoch):
        # Reduce the standard deviation over the epochs
        if std > 0:
            std -= epoch*0.1
        else:
            std = 0
        return std

    # The main training loop function
    def train(self):
        pass


