"""
This class consists of the implementation of advantage actor critic network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import torch.optim as opt



class A2C(object):

    def __init__(self, num_hidden_units, input_dim, num_actions, num_q_val,
                 observation_dim, goal_dim,
                 batch_size, use_cuda, gamma, random_seed,
                 actor_optimizer, critic_optimizer,
                 actor_learning_rate, critic_learning_rate,
                 n_games, n_steps,
                 loss_function, non_conv=True,
                 num_conv_layers=None, num_pool_layers=None,
                 conv_kernel_size=None, img_height=None, img_width=None,
                 input_channels=None):

        self.num_hidden_units = num_hidden_units
        self.input_dim = input_dim
        self.non_conv = non_conv
        self.num_actions = num_actions
        self.num_q = num_q_val
        self.obs_dim = observation_dim
        self.goal_dim = goal_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.cuda = use_cuda
        self.gamma = gamma
        self.n_games = n_games
        self.n_steps = n_steps
        self.random_seed = random_seed
        self.actor_optim = actor_optimizer
        self.critic_optim = critic_optimizer
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.criterion = loss_function

        if self.random_seed is not None:
            self.seed(self.random_seed)

        # Convolution Parameters
        self.num_conv = num_conv_layers
        self.pool = num_pool_layers
        self.im_height = img_height
        self.im_width = img_width
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels


        if non_conv:
            self.actor = ActorNonConvNetwork(num_hidden_layers=self.num_hidden_units, output_action=self.num_actions,
                                             input=self.input_dim)

            self.critic = CriticNonConvNetwork(num_hidden_layers=self.num_hidden_units, output_q_value=self.num_q,
                                               input=self.input_dim, action_dim=self.num_actions)

        else:
            self.actor = ActorNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                      input_channels=self.input_channels, output_action=self.num_actions,
                                      dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                      IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width)
            self.critic = CriticNetwork(num_conv_layers=self.num_conv, conv_kernel_size=self.conv_kernel_size,
                                      input_channels=self.input_channels, action_dim=self.num_actions,
                                      dense_layer=self.num_hidden_units, pool_kernel_size=self.pool,
                                      IMG_HEIGHT=self.im_height, IMG_WIDTH=self.im_width, output_q_value=self.num_q)


        if self.cuda:
            self.to_cuda()

        # Create the optimizers for the actor and critic using the corresponding learning rate
        actor_parameters = self.actor.parameters()
        critic_parameters = self.critic.parameters()

        self.actor_optim = opt.Adam(actor_parameters, lr=self.actor_lr)
        self.critic_optim = opt.Adam(critic_parameters, lr=self.critic_lr)

    def to_cuda(self):
        self.target_actor = self.target_actor.cuda()
        self.target_critic = self.target_critic.cuda()
        self.actor = self.actor.cuda()
        self.critic = self.critic.cuda()

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the actor and critic")
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self, s):
        """
        Setting the random seed for a particular training iteration
        :param s:
        :return:
        """
        torch.manual_seed(s)
        if self.cuda:
            torch.cuda.manual_seed(s)


    def collect_minibatch(self, batch_size):
        """
        Collect a mini-batch of data
        :return: A minibatch of simulations in the open ai gym environment
        """


    # Training procedure
    def train(self):










def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNetwork(nn.Module):
    # The actor network takes the state as input and outputs an action
    # The actor network is used to approximate the argmax action in a continous action space
    # The actor network in the case of a discrete action space is just argmax_a(Q(s,a))

    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_action, dense_layer,
                 pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(ActorNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_kernel = conv_kernel_size
        self.input_channels = input_channels
        self.output_action = output_action
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.im_height = IMG_HEIGHT
        self.im_width = IMG_WIDTH

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn3 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu3 = nn.ReLU(inplace=True)

        # Fully connected layer
        self.fully_connected_layer = nn.Linear(234432, self.dense_layer)
        self.relu4 = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(self.dense_layer, output_action)

        # Weight initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = self.relu4(x)
        out = self.output_layer(x)
        return out


# For non image state space
class ActorNonConvNetwork(nn.Module):
    def __init__(self, num_hidden_layers, output_action, input):
        super(ActorNonConvNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input = input
        self.output_action = output_action
        self.init_w = 3e-3

        #Dense Block
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)
        self.tanh = nn.Tanh()

    def init_weights(self, init_w):
        self.dense_1.weight.data = fanin_init(self.dense_1.weight.data.size())
        self.dense_2.weight.data = fanin_init(self.dense_2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, input):
        x = self.dense_1(input)
        x = self.relu1(x)
        x = self.dense_2(x)
        x = self.relu2(x)
        output = self.output(x)
        output = self.tanh(output)
        return output


class CriticNetwork(nn.Module):

    # The Critic Network basically takes the state and action as the input and outputs a q value
    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_q_value, dense_layer,
                pool_kernel_size, IMG_HEIGHT, IMG_WIDTH, action_dim):
        super(CriticNetwork, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.output_q_value = output_q_value
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.action_dim = action_dim

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.pool_kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.pool_kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu2 = nn.ReLU(inplace=True)
        self.fully_connected_layer = nn.Linear(234432, self.dense_layer)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.dense_layer, output_q_value)

        # Weight initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, states, actions):
        x = self.conv1(states)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = x + actions # Adding the action input
        x = self.relu3(x)
        output = self.output(x)
        return output


class CriticNonConvNetwork(nn.Module):

    def __init__(self, num_hidden_layers, output_q_value, input,
                 action_dim):
        super(CriticNonConvNetwork, self).__init__()
        # Initialize the variables
        self.num_hidden = num_hidden_layers
        self.output_dim = output_q_value
        self.input = input
        self.action_dim = action_dim
        self.init_w = 3e-3

        # Dense Block
        self.dense1 = nn.Linear(self.input, self.num_hidden)
        self.relu1 = nn.ReLU(inplace=True)
        self.hidden2 = nn.Linear(self.num_hidden + self.action_dim, self.num_hidden)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden, self.output_dim)

    def init_weights(self, init_w):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.hidden2.weight.data = fanin_init(self.hidden2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, states, actions):
        #print(states)
        x = self.dense1(states)
        x = self.relu1(x)
        x = torch.cat((x, actions), dim=1)
        x = self.hidden2(x)
        x = self.relu3(x)
        out = self.output(x)
        return out