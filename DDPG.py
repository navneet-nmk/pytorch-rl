# This script contains the Actor and Critic classes
import torch.nn as nn
import math
import torch

class ActorDDPGNetwork(nn.Module):
    # The actor network takes the state as input and outputs an action
    # The actor network is used to approximate the argmax action in a continous action space
    # The actor network in the case of a discrete action space is just argmax_a(Q(s,a))

    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_action, dense_layer,
                 pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(ActorDDPGNetwork, self).__init__()
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
class ActorDDPGNonConvNetwork(nn.Module):
    def __init__(self, num_hidden_layers, output_action, input):
        super(ActorDDPGNonConvNetwork, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input = input
        self.output_action = output_action

        #Dense Block
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
        self.bn1 = nn.BatchNorm1d(num_features=self.num_hidden_layers)
        self.bn1.train(False)
        self.relu1 = nn.ReLU(inplace=True)
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_hidden_layers)
        self.bn2.train(False)
        self.relu2 = nn.ReLU(inplace=True)
        self.dense_3 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)
        self.tanh = nn.Tanh()

        # Weight Initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features*m.out_features
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.dense_1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dense_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dense_3(x)
        x = self.relu3(x)
        output = self.output(x)
        output = self.tanh(output)
        return output


class CriticDDPGNetwork(nn.Module):

    # The Critic Network basically takes the state and action as the input and outputs a q value
    def __init__(self, num_conv_layers, conv_kernel_size, input_channels, output_q_value, dense_layer,
                pool_kernel_size, IMG_HEIGHT, IMG_WIDTH):
        super(CriticDDPGNetwork, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.input_channels = input_channels
        self.output_q_value = output_q_value
        self.dense_layer = dense_layer
        self.pool_kernel_size = pool_kernel_size
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH

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


class CriticDDPGNonConvNetwork(nn.Module):

    def __init__(self, num_hidden_layers, output_q_value, input):
        super(CriticDDPGNonConvNetwork, self).__init__()
        # Initialize the variables
        self.num_hidden = num_hidden_layers
        self.output_dim = output_q_value
        self.input = input

        # Dense Block
        self.dense1 = nn.Linear(self.input, self.num_hidden)
        self.bn1 = nn.BatchNorm1d(self.num_hidden)
        self.relu1 = nn.ReLU(inplace=True)
        self.hidden1 = nn.Linear(self.num_hidden, self.num_hidden)
        self.bn2 = nn.BatchNorm1d(self.num_hidden)
        self.relu2 = nn.ReLU(inplace=True)
        self.hidden2 = nn.Linear(68, self.num_hidden)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(self.num_hidden, self.output_dim)

        # Weight Initialization from a uniform gaussian distribution
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, states, actions):
        x = self.dense1(states)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = torch.cat((x, actions), dim=1)
        x = self.hidden2(x)
        x = self.relu3(x)
        out = self.output(x)
        return out