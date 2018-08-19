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

    def __init__(self, height, width,
                 image_channels, conv_layers,
                 hidden,
                 conv_kernel_size, latent_dimension):
        super(Decoder, self).__init__()
        self.height = height
        self.width = width
        self.out_channels = image_channels
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.in_dimension = latent_dimension
        self.hidden = hidden

        # Decoder Architecture
        self.linear1_decoder = nn.Linear(in_features=self.in_dimension,
                                         out_features=self.hidden)
        self.bn_l_d = nn.BatchNorm1d(self.hidden)
        self.linear = nn.Linear(in_features=self.hidden,
                                out_features=self.height // 16 * self.width // 16 * self.conv_layers * 2)
        self.bn_l_2_d = nn.BatchNorm1d(self.height // 16 * self.width * 16 * self.conv_layers * 2)
        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers * 2,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(self.conv_layers * 2)
        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers * 2,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(self.conv_layers * 2)
        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(self.conv_layers)
        self.conv8 = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.in_channels,
                                         kernel_size=self.conv_kernel_shape - 3, )

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Output activation
        self.output_sigmoid = nn.Sigmoid()

        # Initialize weights using xavier initialization
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear1_decoder.weight)
        nn.init.xavier_uniform_(self.output.weight)
             
    def forward(self, z):
        z = self.linear1_decoder(z)
        z = self.l_relu(z)
        z = self.linear(z)
        z = self.l_relu(z)
        z = z.view((-1, self.conv_layers * 2, self.height // 16, self.width // 16))

        z = self.conv5(z)
        z = self.l_relu(z)

        z = self.conv6(z)
        z = self.l_relu(z)

        z = self.conv7(z)
        z = self.l_relu(z)

        z = self.conv8(z)
        z = self.l_relu(z)

        output = self.output(z)
        output = self.sigmoid_output(output)

        return output


# The inverse dynamic module
class InverseDM(nn.Module):

    def __init__(self):
        super(InverseDM, self).__init__()


class INVAE(nn.Module):

    def __init__(self):
        super(INVAE, self).__init__()