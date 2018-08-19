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
from Layers.Spectral_norm import SpectralNorm

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

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(InverseDM, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Inverse Dynamics Architecture
        self.input_linear = nn.Linear(in_features=self.latent_dim*2, out_features=self.hidden_dim)
        self.hidden_1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.hidden_2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.hidden_3 = nn.Linear(in_features=self.hidden_dim*2, out_features=self.hidden_dim*2)
        self.output = nn.Linear(in_features=self.hidden_dim*2, out_features=self.action_dim)

        # Output Activation function
        self.output_softmax = nn.Softmax()

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.hidden_3.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, current_state, next_state):
        x = torch.cat([current_state, next_state])
        x = self.input_linear(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        x = self.lrelu(x)
        x = self.hidden_3(x)
        x = self.lrelu(x)

        output = self.output(x)
        output = self.output_softmax(output)

        return output


class INVAE(nn.Module):

    def __init__(self, conv_layers,
                 conv_kernel_size, height,
                 width, latent_dim, hidden_dim,
                 input_dim, action_dim,
                 ):
        super(INVAE, self).__init__()

        self.encoder = Encoder(conv_kernel_size=conv_kernel_size, conv_layers=conv_layers,
                               height=height, width=width, in_channels=input_dim,
                               latent_dimension=latent_dim)

        self.decoder = Decoder(conv_layers=conv_layers, conv_kernel_size=conv_kernel_size,
                               latent_dimension=latent_dim, height=height,
                               width=width, hidden=hidden_dim, image_channels=input_dim)

        self.inverse_dm = InverseDM(latent_dim=latent_dim, action_dim=action_dim,
                                    hidden_dim=hidden_dim)

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

    def forward(self, current_state, next_state):

        mu_current, logvar_current = self.encoder(current_state)
        mu_next, logvar_next = self.encoder(next_state)

        z_current = self.reparameterize(mu_current, logvar_current)
        z_next = self.reparameterize(mu_next, logvar_next)

        reconstructed_current_state = self.decoder(z_current)
        reconstructed_next_state = self.decoder(z_next)

        action = self.inverse_dm(z_current, z_next)

        return action, reconstructed_current_state, \
               reconstructed_next_state, mu_current, mu_next, \
               logvar_current, logvar_next, z_current, z_next


# This model takes as input the current state and the action and predicts the next state
class StandardForwardDynamics(nn.Module):

    def __init__(self, action_dim, state_dim, hidden_dim):
        super(StandardForwardDynamics, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Forward dynamics architecture
        self.input_linear = nn.Linear(in_features=self.action_dim+self.state_dim,
                                      out_features=self.hidden_dim)
        self.hidden_1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.output = nn.Linear(in_features=self.hidden_dim*2, out_features=self.state_dim)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, action):
        # Concatenate the state and the action

        # Note that the state in this case is the feature representation of the state

        input = torch.cat([state, action], dim=-1)
        x = self.input_linear(input)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        output = self.output(x)

        return output


# This model takes as input the current state and the action and predicts the next state
# using a infogan which maximizes the information from the action.

# This model consists of 2 networks - Generator and the Discriminator

class Generator(nn.Module):

    """
    The generator/decoder in the CVAE-GAN pipeline

    Given a latent encoding or a noise vector, this network outputs an image.

    """

    def __init__(self, latent_space_dimension,
                 hidden_dim, action_dim):
        super(Generator, self).__init__()

        self.z_dimension = latent_space_dimension
        self.hidden = hidden_dim
        self.action_dim = action_dim

        # We will be using spectral norm in both the generator as well as the discriminator
        # since this improves the training dynamics (https://arxiv.org/abs/1805.08318)

        # Decoder/Generator Architecture

        self.input_linear = SpectralNorm(nn.Linear(in_features=self.action_dim+self.z_dimension,
                                                   out_features=self.hidden))
        self.hidden_1  = SpectralNorm(nn.Linear(in_features=self.hidden, out_features=self.hidden*2))
        self.output = SpectralNorm(nn.Linear(in_features=self.hidden*2, out_features=self.z_dimension))

        self.relu = nn.ReLU(inplace=True)

        # The stability of the GAN Game suffers from the problem of sparse gradients
        # Therefore, try to use LeakyRelu instead of relu
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Use dropouts in the generator to stabilize the training
        self.dropout = nn.Dropout()

        self.sigmoid_output = nn.Sigmoid()

    def forward(self, state, action):
        # Concatenate the state and the action

        # Note that the state in this case is the feature representation of the state

        input = torch.cat([state, action], dim=-1)
        x = self.input_linear(input)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        output = self.output(x)

        return output


class Discriminator_recognizer(nn.Module):

    def __init__(self, latent_space_dimension,
                 hidden_dim, action_dim):
        super(Discriminator_recognizer, self).__init__()

        self.z_dimension = latent_space_dimension
        self.hidden = hidden_dim
        self.action_dim = action_dim

        # Discriminator architecture
        self.input_linear = SpectralNorm(nn.Linear(in_features=self.z_dimension, out_features=self.hidden))
        self.hidden_1 = SpectralNorm(nn.Linear(self.hidden, self.hidden*2))
        self.output = SpectralNorm(nn.Linear(self.hidden*2, 1))
        self.action_output = SpectralNorm(nn.Linear(self.hidden*2, self.action_dim))

        self.relu = nn.ReLU(inplace=True)

        # The stability of the GAN Game suffers from the problem of sparse gradients
        # Therefore, try to use LeakyRelu instead of relu
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.sigmoid_output = nn.Sigmoid()
        self.softmax_output = nn.Softmax()

    def forward(self, state):

        x = self.input_linear(state)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)

        output = self.output(x)
        output = self.sigmoid_output(output)

        actions = self.action_output(x)
        actions = self.softmax_output(actions)

        return output, actions

