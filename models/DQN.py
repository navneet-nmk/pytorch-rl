# The file contains the Convolutional neural network as well as the replay buffer
import torch
import torch.nn as nn
import random
import math
from torch.autograd import Variable
from Memory.Buffer import  ReplayBuffer
from Memory import Buffer
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Environments.super_mario_bros_env as me
import Environments.env_wrappers as env_wrappers
from Utils.utils import to_tensor

USE_CUDA = torch.cuda.is_available()


# Random Encoder
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
                               kernel_size=self.conv_kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers,
                               out_channels=self.conv_layers,
                               kernel_size=self.conv_kernel_size, stride=2, padding=1)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height // 4 * self.width // 4 * self.conv_layers,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.state_space)

        # Initialize the weights of the network (Since this is a random encoder, these weights will
        # remain static during the training of other networks).
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        x = self.conv1(state)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = x.view((-1, self.height//4*self.width//4*self.conv_layers))
        x = self.hidden_1(x)
        x = self.lrelu(x)
        encoded_state = self.output(x)
        return encoded_state


def epsilon_greedy_exploration():
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    return epsilon_by_frame


class DQN(object):
    """
    The Deep Q Network
    """

    def __init__(self,
                 env,
                 num_hidden_units,
                 num_epochs,
                 learning_rate,
                 buffer_size,
                 discount_factor,
                 num_rollouts,
                 num_training_steps,
                 random_seed,
                 state_space,
                 action_space,
                 num_frames,
                 batch_size,
                 height_img,
                 width_img,
                 train_limit_buffer,
                 use_cuda=False,
                 ):
        self.env = env
        self.num_hidden_units = num_hidden_units
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_training_steps = num_training_steps
        self.lr = learning_rate
        self.seed = random_seed
        self.use_cuda = use_cuda
        self.gamma = discount_factor
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.height = height_img
        self.width = width_img
        self.train_limit_buffer = train_limit_buffer

        self.buffer = ReplayBuffer(capacity=buffer_size, seed=random_seed)

        self.current_model = QNetwork(env=self.env, state_space=state_space,
                                      action_space=action_space, hidden=num_hidden_units)
        self.target_model = QNetwork(env=self.env, state_space=state_space,
                                      action_space=action_space, hidden=num_hidden_units)
        self.encoder = Encoder(state_space=state_space, conv_kernel_size=3, conv_layers=32,
                               hidden=64, input_channels=1, height=self.height,
                               width=self.width)

        if self.use_cuda:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optim = optim.Adam(lr=self.lr, params=self.current_model.parameters())

        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    # Calculate the Temporal Difference Error
    def calc_td_error(self):
        """
        Calculates the td error against the bellman target
        :return:
        """
        # Calculate the TD error only for the particular transition

        # Get the separate values from the named tuple
        transitions = self.buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        state = batch.state
        new_state = batch.next_state
        action = batch.action
        reward = batch.reward
        done = batch.done

        #reward = list(reward)
        #done = list(done)

        state = Variable(torch.cat(state), volatile=True)
        new_state = Variable(torch.cat(new_state), volatile=True)
        action = Variable(torch.cat(action))
        reward = Variable(torch.cat(reward))
        done = Variable(torch.cat(done))

        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            new_state = new_state.cuda()
            done = done.cuda()

        q_values = self.current_model(state)
        next_q_values = self.current_model(new_state)
        next_q_state_values = self.target_model(new_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def plot(self, frame_idx, rewards, losses):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        fig.savefig('DQN-SuperMarioBros'+str(frame_idx)+'.jpg')

    # Main training loop
    def train(self):
        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        state = to_tensor(state, use_cuda=self.use_cuda)
        state = self.encoder(state)

        for frame_idx in range(1, self.num_frames+1):
            epsilon_by_frame = epsilon_greedy_exploration()
            epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            next_state, reward, done, success = self.env.step(action.item())
            reward = reward/100
            episode_reward += reward

            next_state = to_tensor(next_state, use_cuda=self.use_cuda)
            next_state = self.encoder(next_state)

            reward = torch.tensor([reward], dtype=torch.float)

            done_bool = done * 1
            done_bool = torch.tensor([done_bool], dtype=torch.float)

            self.buffer.push(state, action, next_state, reward, done_bool, success)

            state = next_state

            if done:
                state = self.env.reset()
                state = to_tensor(state, use_cuda=self.use_cuda)
                state = self.encoder(state)
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.buffer) > self.train_limit_buffer:
                for t in range(self.num_training_steps):
                    loss = self.calc_td_error()
                    losses.append(loss.data[0])

            if frame_idx % 2000 == 0:
                self.plot(frame_idx, all_rewards, losses)
                print('Reward ', str(np.mean(all_rewards)))
                print('Loss', str(np.mean(losses)))

            if frame_idx % 1000 == 0:
                self.update_target_network()


class ConvQNetwork(nn.Module):

    def __init__(self, num_conv_layers, input_channels, output_q_value, pool_kernel_size,
                 kernel_size, dense_layer_features, IM_HEIGHT, IM_WIDTH):
        super(ConvQNetwork, self).__init__()
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

        # Weight initialization using Xavier initialization

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


class QNetwork(nn.Module):
    def __init__(self,
                 env,
                 state_space,
                 action_space,
                 hidden):
        super(QNetwork, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.env = env

        self.layers = nn.Sequential(
            nn.Linear(self.state_space, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.action_space)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state), volatile=True)
            q_value = self.forward(state)
            # Action corresponding to the max Q Value for the state action pairs

            action = q_value.max(1)[1].view(1)
        else:
            action = torch.tensor([random.randrange(self.env.action_space.n)], dtype=torch.long)
        return action


if __name__ == '__main__':
    # Create the mario environment
    env = me.get_mario_bros_env()
    # Add the required environment wrappers
    env = env_wrappers.wrap_wrap(env, height=84, width=84)
    env = env_wrappers.wrap_pytorch(env)

    # Create the DQN Model
    dqn = DQN(env=env, num_hidden_units=128,
              num_epochs=10, learning_rate=1e-3,
              action_space=env.action_space.n, state_space=64,
              batch_size=16, buffer_size=50000, discount_factor=0.99,
              height_img=84, width_img=84, num_frames=100000,
              num_rollouts=10, num_training_steps=10, random_seed=1000000, train_limit_buffer=10000)
    dqn.train()



