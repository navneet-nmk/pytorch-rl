"""

This script contains an implementation of the models for learning an intrinsically motivated
agent trained empowerment.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils.utils import to_tensor
import torch.optim as optim
USE_CUDA = torch.cuda.is_available()


class Encoder(nn.Module):
    
    def __init__(self,
                 state_space,
                 conv_kernel_size,
                 conv_layers,
                 hidden,
                 input_channels,
                 height,
                 width
                 ):
        super(Encoder, self).__init__()
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden
        self.state_space = state_space
        self.input_channels = input_channels
        self.height = height
        self.width = width

        # Random Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers * 2,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height // 16 * self.width // 16 * self.conv_layers * 2,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.state_space)

        # Initialize the weights of the network (Since this is a random encoder, these weights will
        # remain static during the training of other networks).
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        x = self.conv1(state)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        encoded_state = self.output(x)
        return encoded_state


class source_distribution(nn.Module):

    def __init__(self,
                 action_space,
                 conv_kernel_size,
                 conv_layers,
                 hidden, input_channels,
                 height, width,
                 state_space=None,
                 use_encoded_state=True):
        super(source_distribution, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.input_channels = input_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        self.height = height
        self.width = width
        self.use_encoding = use_encoded_state

        # Source Architecture
        # Given a state, this network predicts the action

        if use_encoded_state:
            self.layer1 = nn.Linear(in_features=self.state_space, out_features=self.hidden)
            self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
            self.layer3 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
            self.layer4 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden*2)
            self.hidden_1 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden)
        else:
            self.layer1 = nn.Conv2d(in_channels=self.input_channels,
                                   out_channels=self.conv_layers,
                                   kernel_size=self.conv_kernel_size, stride=2)
            self.layer2 = nn.Conv2d(in_channels=self.conv_layers,
                                   out_channels=self.conv_layers,
                                   kernel_size=self.conv_kernel_size, stride=2)
            self.layer3 = nn.Conv2d(in_channels=self.conv_layers,
                                   out_channels=self.conv_layers*2,
                                   kernel_size=self.conv_kernel_size, stride=2)
            self.layer4 = nn.Conv2d(in_channels=self.conv_layers*2,
                                   out_channels=self.conv_layers*2,
                                   kernel_size=self.conv_kernel_size, stride=2)

            self.hidden_1 = nn.Linear(in_features=self.height // 16 * self.width // 16 * self.conv_layers * 2,
                                      out_features=self.hidden)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers

        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Output activation function
        self.output_activ = nn.Softmax()

    def forward(self, current_state):
        x = self.layer1(current_state)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.layer3(x)
        x = self.lrelu(x)
        x = self.layer4(x)
        x = self.lrelu(x)
        if not self.use_encoding:
            x = x.view((-1, self.height//16*self.width//16*self.conv_layers*2))
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.output(x)
        output = self.output_activ(x)

        return output

class inverse_dynamics_distribution(nn.Module):
    
    def __init__(self, state_space,
                 action_space,
                 height, width,
                 conv_kernel_size,
                 conv_layers, hidden,
                 use_encoding=True):
        super(inverse_dynamics_distribution, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.height = height
        self.width = width
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden
        self.conv_layers = conv_layers
        self.use_encoding = use_encoding

        # Inverse Dynamics Architecture

        # Given the current state and the next state, this network predicts the action

        self.layer1 = nn.Linear(in_features=self.state_space*2, out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.layer3 = nn.Linear(in_features=self.hidden, out_features=self.hidden * 2)
        self.layer4 = nn.Linear(in_features=self.hidden * 2, out_features=self.hidden * 2)
        self.hidden_1 = nn.Linear(in_features=self.hidden * 2, out_features=self.hidden)

        self.conv1 = nn.Conv2d(in_channels=self.input_channels*2,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers * 2,
                               out_channels=self.conv_layers * 2,
                               kernel_size=self.conv_kernel_size, stride=2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height // 16 * self.width // 16 * self.conv_layers * 2,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Output activation function
        self.output_activ = nn.Softmax()

    def forward(self, current_state, next_state):
        state = torch.cat([current_state, next_state], dim=-1)
        if self.use_encoding:
            x = self.layer1(state)
            x = self.lrelu(x)
            x = self.layer2(x)
            x = self.lrelu(x)
            x = self.layer3(x)
            x = self.lrelu(x)
            x = self.layer4(x)
            x = self.lrelu(x)
            x = self.hidden_1(x)
        else:
            x = self.conv1(state)
            x = self.lrelu(x)
            x = self.conv2(x)
            x = self.lrelu(x)
            x = self.conv3(x)
            x = self.lrelu(x)
            x = self.conv4(x)
            x = self.lrelu(x)
            x = x.view((-1, self.height // 16 * self.width // 16 * self.conv_layers * 2))
            x = self.hidden_1(x)

        x = self.lrelu(x)
        x = self.output(x)
        output = self.output_activ(x)

        return output


class forward_dynamics_model(nn.Module):

    def __init__(self, height,
                 width,
                 state_space, action_space,
                 input_channels, conv_kernel_size,
                 conv_layers, hidden,
                 use_encoding=True):
        super(forward_dynamics_model, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.height = height
        self.width = width
        self.conv_kernel_size = conv_kernel_size
        self.hidden = hidden
        self.conv_layers = conv_layers
        self.use_encoding = use_encoding
        self.input_channels = input_channels

        # Forward Dynamics Model Architecture

        # Given the current state and the action, this network predicts the next state

        self.layer1 = nn.Linear(in_features=self.state_space * 2, out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.layer3 = nn.Linear(in_features=self.hidden, out_features=self.hidden * 2)
        self.layer4 = nn.Linear(in_features=self.hidden * 2, out_features=self.hidden * 2)
        self.hidden_1 = nn.Linear(in_features=self.hidden*2,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden+self.action_space, out_features=self.state_space)

    def forward(self, current_state, action):
        x = self.layer1(current_state)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.layer3(x)
        x = self.lrelu(x)
        x = self.layer4(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = torch.cat([x, action], dim=-1)
        output = self.output(x)

        return output


class PolicyNetwork(nn.Module):

    def __init__(self,
                 state_space,
                 action_space,
                 hidden):
        super(PolicyNetwork, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden

        # Policy Architecture
        self.layer1 = nn.Linear(in_features=self.state_space,
                                out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Output activation function
        self.output_activ = nn.Softmax()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        x = self.layer1(state)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.output(x)
        output = self.output_activ(x)

        return output


class StatisticsNetwork(nn.Module):

    def __init__(self, state_space,
                 action_space,
                 hidden, output_dim):
        super(StatisticsNetwork, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.output_dim = output_dim

        # Statistics Network Architecture
        self.layer1 = nn.Linear(in_features=self.state_space+self.action_space,
                                out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, next_state, action):
        s = torch.cat([next_state, action], dim=-1)
        x = self.layer1(s)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        output = self.output(x)
        return output


class EmpowermentTrainer(object):

    def __init__(self,
                 env,
                 encoder,
                 inverse_dynamics,
                 forward_dynamics,
                 source_distribution,
                 statistics_network,
                 policy_network,
                 encoder_lr,
                 inverse_dynamics_lr,
                 forward_dynamics_lr,
                 source_d_lr,
                 stats_lr,
                 policy_lr,
                 num_train_epochs,
                 num_epochs,
                 num_rollouts,
                 model_output_folder,
                 train_encoder=False,
                 use_mine_formulation=True,
                 use_cuda=False):

        self.encoder = encoder
        self.invd = inverse_dynamics
        self.fwd = forward_dynamics
        self.source = source_distribution
        self.stats = statistics_network
        self.use_cuda = use_cuda
        self.policy_network = policy_network
        self.model_output_folder = model_output_folder
        self.use_mine_formulation = use_mine_formulation
        self.env = env
        self.num_epochs = num_epochs
        self.train_epochs = num_train_epochs
        self.num_rollouts = num_rollouts
        self.e_lr = encoder_lr
        self.invd_lr = inverse_dynamics_lr
        self.fwd_lr = forward_dynamics_lr
        self.source_lr = source_d_lr
        self.stats_lr = stats_lr
        self.policy_lr = policy_lr

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.invd = self.invd.cuda()
            self.fwd = self.fwd.cuda()
            self.policy_network = self.policy_network.cuda()
            self.source_distribution = self.source_distribution.cuda()


        # Define the optimizers
        if train_encoder:
            self.e_optim = optim.Adam(params=self.encoder.parameters(), lr=self.e_lr)
        self.invd_optim = optim.Adam(params=self.invd.parameters(), lr=self.invd_lr)
        self.fwd_optim = optim.Adam(params=self.fwd.parameters(), lr=self.fwd_lr)
        self.policy_optim = optim.Adam(params=self.policy_network.parameters(), lr=self.policy_lr)
        self.source_optim = optim.Adam(params=self.source_distribution.parameters(), lr=self.source_lr)

    def train(self):
        pass




