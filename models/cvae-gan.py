import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


class Encoder(nn.Module):

    """
    The encoder network in the cvae-gan pipeline

    Given an image, this network returns the latent encoding
    for the image.

    """

    def __init__(self, conv_layers, conv_kernel_size,
                 latent_space_dim, hidden_dim, use_cuda,
                 height, width, input_channels,pool_kernel_size):
        super(Encoder, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.z_dim = latent_space_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.height = height
        self.width = width
        self.in_channels = input_channels
        self.pool_size = pool_kernel_size

        # Encoder Architecture

        # 1st Stage
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # 2nd Stage
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size)
        self.pool2  = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # Linear Layer
        self.linear1 = nn.Linear(in_features=self.height//4*self.width//4*self.conv_layers*2, out_features=self.hidden_dim)
        self.latent_mu = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.latent_logvar = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.relu = nn.ReLU(inplace=True)


    def encode(self, x):
        # Encoding the input image to the mean and var of the latent distribution
        bs, _, _, _ = x.shape

        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        pool = self.pool(conv2)

        conv3 = self.conv3(pool)
        conv3 = self.relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4)
        pool2 = self.pool2(conv4)

        pool2 = pool2.view((bs, -1))

        linear = self.linear1(pool2)
        linear = self.relu(linear)
        mu = self.latent_mu(linear)
        logvar = self.latent_logvar(linear)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):

    """
    The generator/decoder in the CVAE-GAN pipeline

    Given a latent encoding or a noise vector, this network outputs an image.

    """

    def __init__(self, latent_space_dimension, conv_kernel_size,
                 conv_layers, hidden_dim, height, width, input_channels):
        super(Generator, self).__init__()

        self.z_dimension = latent_space_dimension
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden_dim
        self.height = height
        self.width = width
        self.input_channels = input_channels

        # Decoder/Generator Architecture
        self.linear_decoder = nn.Linear(in_features=self.z_dimension,
                                        out_features=self.hidden)
        self.linear_decoder1 = nn.Linear(in_features=self.hidden,
                                         out_features=self.height//4 * self.width//4 * self.conv_layers*2)

        # Deconvolution layers
        self.conv1 = nn.ConvTranspose2d(in_channels=self.height//4 * self.width//4 * self.conv_layers*2,
                                        out_channels=self.conv_layers*2, kernel_size=self.conv_kernel_size,
                                        stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size)

        self.output = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.input_channels,
                                kernel_size=self.conv_kernel_size-1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, z):
        z  = self.linear_decoder(z)
        z = self.relu(z)
        z = self.linear_decoder1(z)
        z = self.relu(z)

        z =  z.view((-1, self.conv_layers*2, self.height//4, self.width//4))

        z = self.conv1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.relu(z)
        z = self.conv4(z)
        z = self.relu(z)

        output = self.output(z)

        return output


class Discriminator(nn.Module):

    """
    The discriminator network in the CVAEGAN pipeline

    This network distinguishes the fake images from the real

    """

    def __init__(self):
        super(Discriminator, self).__init__()
