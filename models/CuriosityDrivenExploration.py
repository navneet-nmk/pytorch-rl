"""

This script contains the implementation of the model
presented in the paper - ﻿Curiosity-driven Exploration by Self-supervised Prediction


The agent is composed of 2 subsystems -
1. A reward generator which outputs a curiosity driven intrinsic reward
2. A Policy network that outputs a sequence of actions to maximize the reward

Reward Generator Network

Intrinsic Curiosity Module

The reward generator network consists of 2 parts
1. Inverse Dynamics Model
2. Forward Dynamics Model

The inverse dynamics models takes in the current state and
the next state and tries to predict the plausible action taken.

The forward dynamics model takes in the feature representation of a state and the
action and tries to predict the feature representation of the next state.

﻿The inverse model learns a feature space that encodes information
relevant for predicting the agent’s actions only and the forward model
makes predictions in this feature space.

"""

import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()


# Encoder for the states
class Encoder(nn.Module):

    def __init__(self, conv_layers, conv_kernel_size,
                 input_channels, height,
                 width, use_batchnorm=False,
                 ):
        super(Encoder, self).__init__()
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.use_batchnorm = use_batchnorm

        # Encoder Architecture

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv_layers)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               kernel_size=self.conv_kernel_size, stride=2)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Weight initialization

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, input):
        batch_size, _ ,_, _ = input.shape
        x = self.conv1(input)
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

        # Flatten the output
        x = x.view((batch_size, -1))
        return x


# Inverse Dynamics model
class InverseModel(nn.Module):

    def __init__(self, latent_dimension, action_dimension,
                 hidden_dim):

        super(InverseModel, self).__init__()
        self.input_dim = latent_dimension
        self.output_dim = action_dimension
        self.hidden = hidden_dim

        # Inverse Model architecture

        self.linear_1 = nn.Linear(in_features=self.input_dim*2, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Output Activation
        self.softmax = nn.Softmax()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, next_state):

        # Concatenate the state and the next state
        input = torch.cat([state, next_state], dim=-1)
        x = self.linear_1(input)
        x = self.lrelu(x)
        x = self.output(x)

        output = self.softmax(x)
        return output


# Forward Dynamics Model
class ForwardDynamicsModel(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_dim):

        super(ForwardDynamicsModel, self).__init__()

        self.input_dim = state_dim+action_dim
        self.output_dim= state_dim
        self.hidden = hidden_dim


        # Forward Model Architecture

        self.linear_1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.output_dim)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, action):

        # Concatenate the state and the action

        # Note that the state in this case is the feature representation of the state

        input = torch.cat([state, action], dim=-1)
        x = self.linear_1(input)
        x = self.lrelu(x)
        output = self.output(x)

        return output


class IntrinsicCuriosityModule(object):

    def __init__(self,
                 inverse_model,
                 forward_dynamics_model,
                 inverse_lr,
                 forward_lr,
                 num_epochs,
                 save_path):

        self.inverse_model = inverse_model
        self.forward_dynamics_model = forward_dynamics_model
        self.inverse_lr = inverse_lr
        self.forward_lr = forward_lr
        self.save_path = save_path

        self.inverse_optim = optim.Adam(lr=self.inverse_lr, params=self.inverse_model.parameters())
        self.forward_optim = optim.Adam(lr=self.forward_lr, params=self.forward_dynamics_model.parameters())
        self.num_epochs = num_epochs

    def get_inverse_dynamics_loss(self):
        criterionID = nn.BCELoss()
        return criterionID

    def get_forward_dynamics_loss(self):
        criterionFD = nn.MSELoss()
        return criterionFD

    def fit_batch(self, state, action, next_state, train=True):
        # Predict the action from the current state and the next state
        pred_action = self.inverse_model(state, next_state)
        criterionID = self.get_inverse_dynamics_loss()
        inverse_loss = criterionID(pred_action, action)
        if train:
            self.inverse_optim.zero_grad()
            inverse_loss.backward()
            self.inverse_optim.step()

        # Predict the next state from the current state and the action
        pred_next_state = self.forward_dynamics_model(state, action)
        criterionFD = self.get_forward_dynamics_loss()
        forward_loss = criterionFD(pred_next_state, next_state)
        if train:
            self.forward_optim.zero_grad()
            forward_loss.backward()
            self.forward_optim.step()

        return inverse_loss, forward_loss