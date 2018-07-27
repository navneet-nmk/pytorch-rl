import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

USE_CUDA = torch.cuda.is_available()


class InfoGAN(object):
    # The InfoGAN class consisting of the Generator, Discriminator and the Recognizer
    def __init__(self,
                 conv_layers, conv_kernel_size,
                 generator_input_channels, generator_output_channels,
                 height, width, discriminator_input_channels,
                 discriminator_output_dim,
                 output_dim, categorical_dim, continuous_dim,
                 hidden_dim, pool_kernel_size, generator_lr, discriminator_lr):

        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.cat_dim = categorical_dim
        self.cont_dim = continuous_dim
        self.hidden = hidden_dim
        self.height = height
        self.width = width
        self.d_output_dim = discriminator_output_dim
        self.g_input_channels = generator_input_channels
        self.g_output_channels = generator_output_channels
        self.d_input_channels = discriminator_input_channels
        self.output_dim = output_dim



        # Generator
        self.generator = Generator(conv_layers=self.conv_layers, conv_kernel_size=self.conv_kernel_size,
                                   input_channels=self.g_input_channels, output_channels=self.g_output_channels)

        # Discriminator
        self.discriminator = Discriminator_recognizer(conv_layers=self.conv_layers, conv_kernel_size=conv_kernel_size,
                                                      input_dim=self.d_input_channels, height=self.height,
                                                      width=self.width, output_dim=self.output_dim,
                                                      categorical_dim=self.cat_dim, continuous_dim=self.cont_dim,
                                                      discriminator_output_dim=self.d_output_dim, hidden_dim=self.hidden,
                                                      pool_kernel_size=self.pool_kernel_size
                                                      )

        self.gen_optim = Adam(self.generator.parameters(), lr=generator_lr)
        self.dis_optim = Adam(self.discriminator.parameters(), lr=discriminator_lr)

    # Loss Function
    def loss(self):
        pass






class Generator(nn.Module):

    def __init__(self, conv_layers,
                 conv_kernel_size, input_channels, output_channels):
        super(Generator, self).__init__()


        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Generator input -> Noise Vector+Latent Codes

        self.conv1 = nn.ConvTranspose2d(in_channels=self.input_channels,
                               out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv4 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.conv_layers, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size)

        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers,
                                        out_channels=self.output_channels, padding=0, stride=2,
                                        kernel_size=self.conv_kernel_size-2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):
        # Input shape : Noise dimension + latent code dimension
        x = torch.cat((x, c))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)

        x = self.output(x)
        return x


class Discriminator_recognizer(nn.Module):

    # The discriminator and recognizer network for the infogan

    def __init__(self, conv_layers, conv_kernel_size, height, width, input_dim,
                 output_dim, categorical_dim, continuous_dim, pool_kernel_size,
                 hidden_dim, discriminator_output_dim):
        super(Discriminator_recognizer, self).__init__()

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.height = height
        self.width = width
        self.input_channels = input_dim
        self.cat_dim = categorical_dim
        self.cont_dim = continuous_dim
        self.output_dim = output_dim
        self.pool = pool_kernel_size
        self.hidden = hidden_dim
        self.d_output_dim = discriminator_output_dim

        # Shared Network
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=self.conv_layers,
                               padding=0, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               padding=0, kernel_size=self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               padding=0, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               padding=0, kernel_size=self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool)

        self.conv5 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*4,
                               padding=0, kernel_size=self.kernel_size)
        self.conv6 = nn.Conv2d(in_channels=self.conv_layers*4, out_channels=self.conv_layers*4,
                               padding=0, kernel_size=self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool)

        height = self.height//8
        width = self.width//8

        self.linear1 = nn.Linear(in_features=height*width*self.conv_layers*4, out_features=self.hidden)
        self.discriminator_output = nn.Linear(in_features=self.hidden, out_features=self.d_output_dim)
        self.recognizer_output_cont = nn.Linear(in_features=self.hidden, out_features=self.cont_dim)
        self.recognizer_output_cat = nn.Linear(in_features=self.hidden, out_features=self.cat_dim)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, h, w, c = x.shape

        # Input to this network is the output of the generator
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view((b, -1))

        x = self.linear1(x)

        discriminator_output = self.discriminator_output(x)
        recognizer_output_cont = self.recognizer_output_cont(x)
        rcat = self.recognizer_output_cat(x)
        recognizer_output_cat = self.softmax(rcat)

        return discriminator_output, recognizer_output_cont, recognizer_output_cat










