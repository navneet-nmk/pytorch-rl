import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Variational Autoencoder with the option for tuning the disentaglement- Refer to the paper - beta VAE
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
        # Encoding the input image to the mean and var of the latent distribution
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
        # Decoding the image from the latent vector
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


# Definition of the loss function -> Defining beta which is used in beta-vae
def loss_function(recon_x, x, mu, logvar, beta, BATCH_SIZE):
    # This is the log p(x|z) defined as the mean squared loss between the
    # reconstruction and the original image
    MSE = nn.MSELoss(x, recon_x)

    # KLD - Kullback liebler divergence -- how much one learned distribution
    # deviate from one another, in this case the learned distribution
    # from the unit Gaussian.

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by the same number of elements in reconstruction
    KLD = KLD/BATCH_SIZE

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian

    # To learn disentangled representations, we use the beta parameter
    # as in the beta-vae
    loss = MSE + beta*KLD

    return loss


# Denoising Autoencoder
class DAE(nn.Module):
    def __init__(self, conv_layers,
                 conv_kernel_size, pool_kernel_size,
                 height, width, input_channels,
                 activation, hidden_dim
                 ):
        super(DAE, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_shape = conv_kernel_size
        self.pool = pool_kernel_size
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.activ = activation
        self.hidden = hidden_dim


        # Encoder
        # ﻿four convolutional layers, each with kernel size 4 and stride 2 in both the height and width dimensions.
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_shape, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_shape, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_layers)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_shape, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_shape, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*2)

        self.relu = nn.ReLU(inplace=True)

        # Bottleneck Layer
        self.bottleneck = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*2,
                                    out_features=self.hidden)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(in_channels=self.hidden,
                                        out_channels=self.conv_layers*2, stride=2, kernel_size=self.conv_kernel_shape
                                        ,padding=1)
        self.bn5 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers*2,
                                        out_channels=self.conv_layers * 2, stride=2, kernel_size=self.conv_kernel_shape
                                        , padding=1)
        self.bn6 = nn.BatchNorm2d(self.conv_layers * 2)
        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2,
                                        out_channels=self.conv_layers, stride=2, kernel_size=self.conv_kernel_shape
                                        , padding=1)
        self.bn7 = nn.BatchNorm2d(self.conv_layers)
        self.conv8 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, stride=2, kernel_size=self.conv_kernel_shape
                                        , padding=1)
        self.bn8 = nn.BatchNorm2d(self.conv_layers)
        # Decoder output
        self.output = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.input_channels,
                                kernel_size=self.conv_kernel_shape, padding=1)


    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        out = self.bottleneck(x)
        return out
    
    def decode(self, encoded):
        x = self.conv5(encoded)
        x= self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        out = self.output(x)
        return out

    def forward(self, image):
        encoded = self.encode(image)
        decoded = self.decode(image)
        return decoded, encoded




