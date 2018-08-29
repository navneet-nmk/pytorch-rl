import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Variational Autoencoder with the option for tuning the disentaglement- Refer to the paper - beta VAE
class VAE(nn.Module):
    def __init__(self, conv_layers, z_dimension, pool_kernel_size,
                 conv_kernel_size, input_channels, height, width, hidden_dim, use_cuda, use_skip_connections=True):
        super(VAE, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_shape = conv_kernel_size
        self.pool = pool_kernel_size
        self.z_dimension = z_dimension
        self.in_channels = input_channels
        self.height = height
        self.width = width
        self.hidden = hidden_dim
        self.use_cuda = use_cuda
        self.use_skip = use_skip_connections

        # Intialize a list of skip values to be used for the decoder
        skip_layers  = ['conv1', 'conv2', 'conv3', 'conv4']
        self.skip_values = dict(skip_layers)


        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
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
        # Size of input features = HxWx2C
        self.linear1 = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*2, out_features=self.hidden)
        self.bn_l = nn.BatchNorm1d(self.hidden)
        self.latent_mu = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.latent_logvar = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Architecture
        self.linear1_decoder = nn.Linear(in_features=self.z_dimension,
                                         out_features=self.hidden)
        self.bn_l_d = nn.BatchNorm1d(self.hidden)
        self.linear = nn.Linear(in_features=self.hidden, out_features=self.height//16*self.width//16*self.conv_layers*2)
        self.bn_l_2_d  =nn.BatchNorm1d(self.height//16*self.width*16*self.conv_layers*2)
        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv6  = nn.ConvTranspose2d(in_channels=self.conv_layers*2,  out_channels=self.conv_layers*2,
                                         kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(self.conv_layers)
        self.conv8 = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.in_channels,
                                        kernel_size=self.conv_kernel_shape-3,)

        # Define the leaky relu activation function
        self.l_relu = nn.LeakyReLU(0.1)

        # Output Activation function
        self.sigmoid_output = nn.Sigmoid()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear1_decoder.weight)
        nn.init.xavier_uniform_(self.latent_mu.weight)
        nn.init.xavier_uniform_(self.latent_logvar.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def encode(self, x):
        # Encoding the input image to the mean and var of the latent distribution
        bs, _, _, _ = x.shape
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.l_relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.l_relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.l_relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.l_relu(conv4)

        fl = conv4.view((bs, -1))

        linear = self.linear1(fl)
        linear = self.bn_l(linear)
        linear = self.l_relu(linear)
        mu = self.latent_mu(linear)
        logvar = self.latent_logvar(linear)

        self.skip_values['conv1'] = conv1
        self.skip_values['conv2'] = conv2
        self.skip_values['conv3'] = conv3
        self.skip_values['conv4'] = conv4

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # Decoding the image from the latent vector
        z = self.linear1_decoder(z)
        z = self.l_relu(z)
        z = self.linear(z)
        z = self.l_relu(z)
        z = z.view((-1, self.conv_layers*2, self.height//16, self.width//16))
        z = self.conv5(z)
        z = self.l_relu(z)
        # Add skip connections
        z = torch.cat([z, self.skip_values['conv3']])

        z = self.conv6(z)
        z = self.l_relu(z)
        # Add skip connections
        z = torch.cat([z, self.skip_values['conv2']])

        z = self.conv7(z)
        z = self.l_relu(z)
        # Add skip connections
        z = torch.cat([z, self.skip_values['conv1']])

        z = self.conv8(z)
        z = self.l_relu(z)

        output = self.output(z)
        output = self.sigmoid_output(output)

        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar, z


# Denoising Autoencoder
class DAE(nn.Module):
    def __init__(self, conv_layers,
                 conv_kernel_size, pool_kernel_size, use_cuda,
                 height, width, input_channels, hidden_dim,
                 noise_scale=0.1):
        super(DAE, self).__init__()

        self.conv_layers = conv_layers
        self.conv_kernel_shape = conv_kernel_size
        self.pool = pool_kernel_size
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.hidden = hidden_dim
        self.noise_scale = noise_scale
        self.cuda_avbl = use_cuda


        # Encoder
        # ﻿four convolutional layers, each with kernel size 4 and stride 2 in both the height and width dimensions.
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_shape, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_shape,  stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_shape, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_shape, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Bottleneck Layer
        self.bottleneck = nn.Linear(in_features=self.height//16*self.width//16*self.conv_layers*2,
                                    out_features=self.hidden)


        # Decoder
        self.linear_decoder = nn.Linear(in_features=hidden_dim, out_features=self.height//16*self.width//16*self.conv_layers*2)
        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers*2,
                                        out_channels=self.conv_layers*2, stride=2, kernel_size=self.conv_kernel_shape-1
                                        )

        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers*2,
                                        out_channels=self.conv_layers * 2, stride=2, kernel_size=self.conv_kernel_shape-1
                                        )

        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2,
                                        out_channels=self.conv_layers, stride=2, kernel_size=self.conv_kernel_shape-1
                                        )

        self.conv8 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, stride=2, kernel_size=self.conv_kernel_shape-1
                                        )
        # Decoder output
        self.output = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.input_channels,
                                kernel_size=self.conv_kernel_shape-2)

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = x.view((-1, self.height//16*self.width//16*self.conv_layers*2))

        out = self.bottleneck(x)
        return out
    
    def decode(self, encoded):
        x = self.linear_decoder(encoded)
        x  = x.view((-1, self.conv_layers*2, self.height//16, self.width//16))
        x = self.conv5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.relu(x)

        out = self.output(x)
        return out

    def forward(self, image):
        # Adding noise
        n, _, _, _ = image.shape
        noise = Variable(torch.randn(n, 3, self.height, self.width))
        if self.cuda_avbl:
            noise = noise.cuda()
        #image = torch.mul(image + 0.25, 0.1 * noise)
        image = torch.add(image, self.noise_scale*noise)
        encoded = self.encode(image)
        decoded = self.decode(encoded)
        return decoded, encoded