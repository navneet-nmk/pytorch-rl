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

USE_CUDA = torch.cuda.is_available()


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

        self.buffer = ReplayBuffer(capacity=buffer_size, seed=random_seed)

        self.current_model = QNetwork(env=self.env, state_space=state_space,
                                      action_space=action_space, hidden=num_hidden_units)
        self.target_model = QNetwork(env=self.env, state_space=state_space,
                                      action_space=action_space, hidden=num_hidden_units)

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

        state = Variable(state)
        new_state = Variable(new_state)
        reward = Variable(reward)
        action = Variable(action)
        done = Variable(done)

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
        fig.savefig('DQN-SuperMarioBros.jpg')

    # Main training loop
    def train(self):
        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()

        for frame_idx in range(1, self.num_frames+1):
            epsilon_by_frame = epsilon_greedy_exploration()
            epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            next_state, reward, done, success = self.env.step(action)
            self.buffer.push(state, action, next_state, reward, done, success)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.buffer) > self.batch_size:
                loss = self.calc_td_error()
                losses.append(loss.data[0])

            if frame_idx % 200 == 0:
                self.plot(frame_idx, all_rewards, losses)

            if frame_idx % 100 == 0:
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
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            # Action corresponding to the max Q Value for the state action pairs
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action


if __name__ == '__main__':
    


