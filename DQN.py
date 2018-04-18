# The file contains the Convolutional neural network as well as the replay buffer
import torch
import torch.nn as nn
import random
import math

class DQN(object):
    """
    The Deep Q Network
    """

    def __init__(self, num_hidden_units):
        self.num_hidden_units = num_hidden_units


class ActionPredictionNetwork(nn.Module):

    def __init__(self, num_conv_layers, input_channels, output_q_value, pool_kernel_size, kernel_size, dense_layer_features, IM_HEIGHT, IM_WIDTH):
        super(ActionPredictionNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.input_channels = input_channels
        self.output = output_q_value
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size =  kernel_size
        self.dense_features = dense_layer_features
        self.height = IM_HEIGHT
        self.width = IM_WIDTH

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_conv_layers, padding=0,
                               kernel_size=self.kernel_size)
        self.bn1  = nn.BatchNorm2d(num_features=num_conv_layers)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_conv_layers, out_channels=num_conv_layers*2, padding=0,
                               kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=num_conv_layers*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=num_conv_layers*2, out_channels=num_conv_layers*2, padding=0,
                               kernel_size=self.kernel_size)
        self.bn3 = nn.BatchNorm2d(num_features=num_conv_layers*2)
        self.relu3 = nn.ReLU(inplace=True)

        #Fully connected layer
        self.fully_connected_layer = nn.Linear(234432, self.dense_features)
        self.relu4 = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(256, output_q_value)

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


class ActionPredNonConvNetwork(nn.Module):
    def __init__(self):
        super(ActionPredNonConvNetwork, self).__init__()


