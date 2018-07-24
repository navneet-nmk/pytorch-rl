import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, conv_layers, z_dimension, pool_kernel_size, activation,
                 conv_kernel_size, input_channels, height, width, hidden_dim):
        super(VAE, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_shape = conv_kernel_size
        self.pool = pool_kernel_size
        self.activ = activation
        self.z_dimension = z_dimension
        self.in_channels = input_channels
        self.height = height
        self.width = width
        self.hidden = hidden_dim

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_shape, padding=1)
        self.bn1 =  nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_shape, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv_layers*2)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)
        # Size of input features = HxWx2C
        self.linear1 = nn.Linear(in_features=self.height//2*self.width//2*self.conv_layers*2, out_features=self.hidden)
        self.bn3 = nn.BatchNorm1d(self.hidden)
        self.latent_mu = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.latent_logvar = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.relu = nn.ReLU(inplace=True)


        # Decoder Architecture
        self.linear1_decoder = nn.Linear(in_features=self.z_dimension,
                                         out_features=self.height*self.width*self.conv_layers*2)
        self.bn4 = nn.BatchNorm1d(self.height*self.width*self.conv_layers*2)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(self.conv_layers)
        self.conv4 = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, padding=1)
        self.bn6 = nn.BatchNorm2d(self.conv_layers)
        self.conv_output = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                              out_channels=self.in_channels,
                                              kernel_size=self.conv_kernel_shape,padding=1)


    def encode(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu(conv2)
        pool = self.pool1(conv2)

        linear = self.linear1(pool)
        linear = self.bn3(linear)
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

    def decode(self, z):
        z = self.linear1_decoder(z)
        z = self.bn4(z)
        z = self.conv3(z)
        z = self.bn5(z)
        z = self.relu(z)
        z = self.conv4(z)
        z = self.bn6(z)
        z = self.relu(z)
        output = self.conv_output(z)
        return output

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    










