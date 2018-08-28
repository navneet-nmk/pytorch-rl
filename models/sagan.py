"""

This file contains the implementation of the Self attention GAN

https://arxiv.org/abs/1805.08318

"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from multiprocessing import Value

import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

USE_CUDA = torch.cuda.is_available()


class SAGAN(object):

    def __init__(self, generator, discriminator,
                 dataset, num_epochs,
                 random_seed, shuffle, use_cuda,
                 tensorboard_summary_writer,
                 output_folder, image_size,
                 image_channels, noise_dim,
                 images_dir, save_iter,
                 generator_lr, discriminator_lr, batch_size,
                 num_threads, num_train_threads,
                 save_images_row=8):

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
        self.image_size = image_size
        self.img_channels = image_channels
        self.noise_dim = noise_dim
        self.images_dir = images_dir
        self.save_iter = save_iter
        self.save_images_row = save_images_row
        self.num_threads = num_threads
        self.num_train_threads = num_train_threads

        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        # Use of Lambda function since the Generator and Discriminator uses spectral norm
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

    def train_multithreaded(self):
        pass

    # The main training loop function
    def train(self):
        real_x = torch.FloatTensor(self.batch_size, self.img_channels,
                                   self.image_size, self.image_size)
        labels = torch.FloatTensor(self.batch_size)
        noise = torch.FloatTensor(self.batch_size, self.noise_dim)

        noise = Variable(noise)
        labels = Variable(labels)
        labels.requires_grad = False

        criterionD = self.disc_loss()

        # For inference
        fix_noise = torch.Tensor(self.noise_dim).uniform_(-1, 1)

        # Main training loop
        for epoch in range(self.num_epochs):
            std = 1.0
            for num_iters, batch_data in enumerate(self.get_dataloader()):
                # Real Part
                self.dis_optim.zero_grad()

                x = batch_data['image']
                bs = x.size(0)

                x = Variable(x)

                if self.use_cuda:
                    x = x.cuda()
                    real_x = real_x.cuda()
                    noise = noise.cuda()
                    labels = labels.cuda()

                real_x.data.copy_(x)
                # Add noise to the inputs of the discriminator

                # This is an auxillary noise added to the discriminator to stabilize the training
                noise_data = torch.zeros(x.shape)
                #            print(noise.shape)
                noise_data = torch.normal(mean=noise_data, std=std)
                if self.use_cuda:
                    noise_data = noise_data.cuda()

                x += noise_data
                d_output = self.discriminator(x)
                labels.data.fill_(1)
                loss_real = criterionD(d_output, labels)
                loss_real.backward()

                # Fake Part
                z = self._noise_sample(noise, bs)
                fake_x = self.generator(z)

                fake_x = fake_x + noise_data
                d_output = self.discriminator(fake_x.detach())
                labels.data.fill_(0)
                loss_fake = criterionD(d_output, labels)
                loss_fake.backward()

                D_loss = loss_real + loss_fake
                self.dis_optim.step()

                # Generator Loss update
                d_output = self.discriminator(fake_x)
                labels.data.fill_(1.0)
                reconstruct_loss = criterionD(d_output, labels)

                self.gen_optim.zero_grad()

                G_loss = reconstruct_loss
                G_loss.backward()

                self.gen_optim.step()

                if num_iters % self.save_iter == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )
                    # Anneal the standard deviation of the noise vector
                    std = self.linear_annealing_variance(std=std, epoch=epoch)

                    noise.data.copy_(fix_noise)

                    z = noise
                    x_save = self.generator(z)
                    save_image(x_save.data.cpu(), self.images_dir + str(epoch)+'c1.png', nrow=self.save_images_row)

            # Save the model at the end of each epoch
            self.save_model(output=self.output_folder)

    def to_cuda(self):
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

    def save_model(self, output):
        print("Saving the generator and discriminator")
        torch.save(
                self.generator.state_dict(),
                '{}/generator.pt'.format(output)
            )
        torch.save(
                    self.discriminator.state_dict(),
                    '{}/discriminator.pt'.format(output)
                )