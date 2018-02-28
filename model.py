# The file contains the Convolutional neural network as well as the replay buffer
import torch
import torch.nn as nn
import random


class ActionPredictionNetwork(nn.Module):

    def __init__(self, num_conv_layers, input_channels, output_q_value, pool_kernel_size, kernel_size, IM_HEIGHT, IM_WIDTH):
        super(ActionPredictionNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.input_channels = input_channels
        self.output = output_q_value
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size =  kernel_size
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
        self.fully_connected_layer = nn.Linear(234432, output_q_value)

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
        out = self.fully_connected_layer(x)

        return out


# Replay buffer which acts similar to a ring queue
class ReplayBuffer(object):

    def __init__(self, size_of_buffer):
        super(ReplayBuffer, self).__init__()
        self.size = size_of_buffer
        self.buffer = []

    def add(self, k):
        if len(self.buffer) < self.size:
            self.buffer.append(k)
        else:
            self.buffer.pop(0)
            self.buffer.append(k)

    def get_buffer_size(self):
        return len(self.buffer)

    def sample_batch(self, sample_size):
        random_sample = [self.buffer[i] for i in sorted(random.sample(range(len(self.buffer)), sample_size))]
        return random_sample