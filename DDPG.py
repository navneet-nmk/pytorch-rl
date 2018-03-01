import torch.nn as nn
import math


class ActorDDPGNetwork(nn.Module):

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
        self.fully_connected_layer = nn.Linear(234432, self.dense_features)
        self.relu4 = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(256, output_action)

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