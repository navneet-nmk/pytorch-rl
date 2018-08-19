"""

This script contains an implementation of the inverse dynamics variational autoencoder.
This basically combines the inverse dynamics model (predicting the action from the
current and the next state) and the variational autoencoder trained with reconstruction error.
As an added parameter, beta is included with the KL divergence inorder to encourage
disentangled representations.

"""

import torch
import torch.nn as nn
from torch.autograd import Variable


# The encoder for the INVAE
class Encoder(nn.Module):

    def __init__(self, conv_layers, conv_kernel_size,
                 latent_dimension, in_channels, height, width):
        super(Encoder, self).__init__()
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.latent_dim = latent_dimension
        self.in_channels = in_channels
        self.height = height
        self.width = width

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn2 = nn.BatchNorm2d(self.conv_layers)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*2)

        #Latent variable mean and logvariance
        self.mu = nn.Linear(in_features=self.height//16*self.width/16*self.conv_layers*2, out_features=self.latent_dim)
        self.logvar = nn.Linear(in_features=self.height//16*self.width/16*self.conv_layers*2, out_features=self.latent_dim)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.logvar.weight)
          
    def forward(self, state):
        x = self.conv1(state)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = x.view((-1, self.height//16*self.width//16*self.conv_layers*2))

        mean = self.mu(x)
        logvar = self.logvar(x)

        return mean, logvar


# The decoder for the INVAE
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()


# The inverse dynamic module
class InverseDM(nn.Module):

    def __init__(self):
        super(InverseDM, self).__init__()


class INVAE(nn.Module):

    def __init__(self):
        super(INVAE, self).__init__()