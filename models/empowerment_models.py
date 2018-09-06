"""

This script contains an implementation of the models for learning an intrinsically motivated
agent trained empowerment. The policy is trained using deep q learning.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils.utils import *
from Memory import Buffer
import torch.optim as optim
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from collections import deque, defaultdict
import time
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import gym
import retro

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


class MDN_LSTM(nn.Module):

    """

    This follows the Mixture Density Network LSTM defined in the paper
    World Models, Ha et al.

    """

    def __init__(self, state_space,
                 action_space,
                 lstm_hidden,
                 gaussians,
                 num_lstm_layers):
        super(MDN_LSTM, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.lstm_hidden = lstm_hidden
        self.gaussians = gaussians
        self.num_lstm_layers = num_lstm_layers

        #self.sequence_length = sequence_length

        self.hidden = self.init_hidden(self.sequence_length)

        # Define the RNN
        self.rnn = nn.LSTM(self.state_space+self.action_space, self.lstm_hidden, self.num_lstm_layers)

        # Define the fully connected layer
        self.s_pi = nn.Linear(self.lstm_hidden, self.state_space*self.gaussians)
        self.s_sigma = nn.Linear(self.lstm_hidden, self.state_space*self.gaussians)
        self.s_mean = nn.Linear(self.lstm_hidden, self.state_space*self.gaussians)

    def init_hidden(self, sequence):
        hidden = torch.zeros(self.num_lstm_layers, sequence, self.lstm_hidden)
        cell = torch.zeros(self.num_lstm_layers, sequence, self.lstm_hidden)
        return hidden, cell

    def forward(self, states, actions):
        self.rnn.flatten_parameters()
        seq_length = actions.size()[1]
        self.hidden = self.init_hidden(seq_length)

        inputs = torch.cat([states, actions], dim=-1)
        s, self.hidden = self.rnn(inputs, self.hidden)

        pi = self.s_pi(s).view(-1, seq_length, self.gaussians, self.state_space)
        pi = F.softmax(pi, dim=2)

        sigma = torch.exp(self.s_sigma(s)).view(-1, seq_length,
                                                self.gaussians, self.state_space)

        mu = self.s_mean(s).view(-1, seq_length, self.gaussians, self.state_space)

        return pi, sigma, mu


class forward_dynamics_lstm(object):

    def __init__(self,
                 sequence_length,
                 state_space,
                 epsilon,
                 mdn_lstm,
                 num_epochs,
                 learning_rate,
                 print_epoch=5
                 ):
        self.seq = sequence_length
        self.state_space = state_space
        self.eps = epsilon
        self.model = mdn_lstm
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.optimizer = optim.Adam(lr=self.lr, params=self.model.parameters())
        self.p_epoch = print_epoch

    def mdn_loss_function(self, out_pi, out_sigma, out_mu, target_next_states):
        y = target_next_states.view(-1, self.seq, 1, self.state_space)
        result = Normal(loc=out_mu, scale=out_sigma)
        result = torch.exp(result.log_prob(y))
        result = torch.sum(result * out_pi, dim=2)
        result = -torch.log(self.eps + result)
        return torch.mean(result)

    def train_on_batch(self, states, actions, next_states):
        for epoch in range(self.num_epochs):
            pis, sigmas, mus = self.model(states, actions)
            loss = self.mdn_loss_function(out_pi=pis, out_sigma=sigmas, out_mu=mus, target_next_states=next_states)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.p_epoch == 0:
                print('LSTM Loss: ', loss)

    def sample_next_state(self, state, action):
        pis, sigmas, mus = self.model(state, action)
        mixt = Categorical(torch.exp(pis)).sample().item()

        next_state = mus[:, mixt, :]

        return next_state


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


class DQN(nn.Module):

    def __init__(self,
                 action_space,
                 hidden,
                 input_channels):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.in_channels = input_channels

        # DQN Architecture
        self.layer1 = nn.Linear(in_features=self.in_channels, out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.action_space)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU(inplace=True)

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
        return x.view((state.size(0), -1))


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
                 target_policy_network,
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
                 size_replay_buffer,
                 size_dqn_replay_buffer,
                 random_seed,
                 polyak_constant,
                 discount_factor,
                 batch_size,
                 action_space,
                 observation_space,
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
        self.target_policy_network = target_policy_network
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
        self.random_seed = random_seed
        self.replay_buffer = Buffer.ReplayBuffer(capacity=size_replay_buffer,
                                                 seed=self.random_seed)
        self.dqn_replay_buffer = Buffer.ReplayBuffer(capacity=size_dqn_replay_buffer,
                                                     seed=self.random_seed)
        self.tau = polyak_constant
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.action_space = action_space
        self.obs_space = observation_space

        torch.manual_seed(self.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.random_seed)

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
        self.stats_optim = optim.Adam(params=self.stats.parameters(), lr=self.stats_lr)

    def get_all_actions(self, action_space):
        all_actions = []
        for i in range(action_space):
            actions = torch.zeros(action_space)
            actions[i] = 1
            all_actions.append(actions)
        return all_actions

    # Store the transition into the replay buffer
    def store_transition(self, buffer, state, new_state, action, reward, done, success):
        buffer.push(state, action, new_state, reward, done, success)

    # Update the networks using polyak averaging
    def update_networks(self):
        for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    # Calculate the temporal difference error
    def calc_td_error(self, transition):
        """
                Calculates the td error against the bellman target
                :return:
                """
        # Calculate the TD error only for the particular transition

        # Get the separate values from the named tuple

        state, new_state, reward, success, action, done = transition

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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_network(state).gather(1, action)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.target_policy_network(new_state).max(1)[0].detach()
        next_state_values = next_state_values * (1 - done)
        y = reward + self.gamma * next_state_values
        td_loss = F.smooth_l1_loss(state_action_values, y)
        return td_loss

    # Train the policy network
    def fit_batch_dqn(self):
        # Sample mini-batch from the replay buffer uniformly or from the prioritized experience replay.

        # If the size of the buffer is less than batch size then return
        if self.replay_buffer.get_buffer_size() < self.batch_size:
            return None

        transitions = self.dqn_replay_buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states), volatile=True)
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        # Encode the states and the new states
        states = self.encoder(states)
        new_states = self.encoder(new_states)
        state_action_values = self.policy_network(states).gather(1, actions)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.target_policy_network(new_states).max(1)[0].detach()
        next_state_values = next_state_values * (1 - dones)
        y = rewards + self.gamma * next_state_values
        td_loss = F.smooth_l1_loss(state_action_values, y)

        self.policy_optim.zero_grad()
        td_loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optim.step()

        return td_loss

    def train_forward_dynamics(self):

        if self.replay_buffer.get_buffer_size() < self.batch_size:
            return None

        transitions = self.replay_buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states), volatile=True)
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        predicted_new_states = self.fwd(states, actions)
        mse_error = F.mse_loss(predicted_new_states, new_states)
        self.fwd_optim.zero_grad()
        mse_error.backward()
        self.fwd_optim.step()

        return mse_error

    def train_statistics_network(self):

        if self.replay_buffer.get_buffer_size() < self.batch_size:
            return None

        transitions = self.replay_buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states), volatile=True)
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        all_actions = self.get_all_actions(self.action_space)
        all_actions = Variable(torch.cat(all_actions))

        new_state_marginals = []
        for state in states:
            state = state.expand(self.action_space, -1)
            new_states = self.fwd(state, all_actions)
            new_states = torch.mean(new_states)
            new_state_marginals.append(new_states)

        new_state_marginals = Variable(torch.cat(new_state_marginals))

        mutual_information = self.stats(new_states, actions) - \
                             torch.log(torch.exp(self.stats(new_state_marginals, actions)))

        # Maximize the mutual information
        loss = -mutual_information
        self.stats_optim.zero_grad()
        loss.backward()
        self.stats_optim.step()

        # Store in the dqn replay buffer

        rewards = rewards + mutual_information
        self.store_transition(buffer=self.dqn_replay_buffer,
                              state=states,
                              action=actions,
                              new_state=new_states,
                              reward=rewards,
                              done=dones, success=None)

        return loss


    def train(self):
        # Starting time
        start_time = time.time()

        # Initialize the statistics dictionary
        statistics = self.statistics

        episode_rewards_history = deque(maxlen=100)
        eval_episode_rewards_history = deque(maxlen=100)
        episode_success_history = deque(maxlen=100)
        eval_episode_success_history = deque(maxlen=100)

        epoch_episode_rewards = []
        epoch_episode_success = []
        epoch_episode_steps = []

        # Epoch Rewards and success
        epoch_rewards = []
        epoch_success = []

        # Initialize the training with an initial state
        state = self.env.reset()

        # If eval, initialize the evaluation with an initial state
        if self.eval_env is not None:
            eval_state = self.eval_env.reset()
            eval_state = to_tensor(eval_state, use_cuda=self.cuda)
            eval_state = torch.unsqueeze(eval_state, dim=0)

        # Initialize the losses
        loss = 0
        episode_reward = 0
        episode_success = 0
        episode_step = 0
        epoch_actions = []
        t = 0

        # Check whether to use cuda or not
        state = to_tensor(state, use_cuda=self.cuda)
        state = torch.unsqueeze(state, dim=0)

        # Main training loop
        for epoch in range(self.num_epochs):
            epoch_actor_losses = []
            epoch_critic_losses = []
            for episode in range(self.max_episodes):

                # Rollout of trajectory to fill the replay buffer before training
                for rollout in range(self.num_rollouts):
                    # Sample an action from behavioural policy pi
                    action = self.ddpg.get_action(state=state, noise=True)
                    assert action.shape == self.env.get_action_shape

                    # Execute next action
                    new_state, reward, done, success = self.env.step(action)
                    success = success['is_success']
                    done_bool = done * 1

                    t += 1
                    episode_reward += reward
                    episode_step += 1
                    episode_success += success

                    # Book keeping
                    epoch_actions.append(action)
                    # Store the transition in the replay buffer of the agent
                    self.store_transition(state=state, new_state=new_state,
                                               action=action, done=done_bool, reward=reward,
                                               success=success)
                    # Set the current state as the next state
                    state = to_tensor(new_state, use_cuda=self.cuda)
                    state = torch.unsqueeze(state, dim=0)

                    # End of the episode
                    if done:
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        episode_success_history.append(episode_success)
                        epoch_episode_success.append(episode_success)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0
                        episode_step = 0
                        episode_success = 0

                        # Get a new initial state to start from
                        state = self.env.reset()
                        state = to_tensor(state, use_cuda=self.use_cuda)

                # Train
                for train_steps in range(self.train_epochs):
                    loss = self.fit_batch()

                    # Update the target networks using polyak averaging
                    self.update_target_network()

                eval_episode_rewards = []
                eval_episode_successes = []
                if self.eval_env is not None:
                    eval_episode_reward = 0
                    eval_episode_success = 0
                    for t_rollout in range(self.num_eval_rollouts):
                        if eval_state is not None:
                            eval_action = self.ddpg.get_action(state=eval_state, noise=False)
                        eval_new_state, eval_reward, eval_done, eval_success = self.eval_env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_episode_success += eval_success

                        if eval_done:
                            eval_state = self.eval_env.reset()
                            eval_state = to_tensor(eval_state, use_cuda=self.cuda)
                            eval_state = torch.unsqueeze(eval_state, dim=0)
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_successes.append(eval_episode_success)
                            eval_episode_success_history.append(eval_episode_success)
                            eval_episode_reward = 0
                            eval_episode_success = 0

            # Log stats
            duration = time.time() - start_time
            statistics['rollout/rewards'] = np.mean(epoch_episode_rewards)
            statistics['rollout/rewards_history'] = np.mean(episode_rewards_history)
            statistics['rollout/successes'] = np.mean(epoch_episode_success)
            statistics['rollout/successes_history'] = np.mean(episode_success_history)
            statistics['rollout/actions_mean'] = np.mean(epoch_actions)
            statistics['train/loss_actor'] = np.mean(epoch_actor_losses)
            statistics['train/loss_critic'] = np.mean(epoch_critic_losses)
            statistics['total/duration'] = duration

            # Evaluation statistics
            if self.eval_env is not None:
                statistics['eval/rewards'] = np.mean(eval_episode_rewards)
                statistics['eval/rewards_history'] = np.mean(eval_episode_rewards_history)
                statistics['eval/successes'] = np.mean(eval_episode_successes)
                statistics['eval/success_history'] = np.mean(eval_episode_success_history)

            # Print the statistics
            if self.verbose:
                if epoch % 5 == 0:
                    print("Actor Loss: ", statistics['train/loss_actor'])
                    print("Critic Loss: ", statistics['train/loss_critic'])
                    print("Reward ", statistics['rollout/rewards'])
                    print("Successes ", statistics['rollout/successes'])

                    if self.eval_env is not None:
                        print("Evaluation Reward ", statistics['eval/rewards'])
                        print("Evaluation Successes ", statistics['eval/successes'])

            # Log the combined statistics for all epochs
            for key in sorted(statistics.keys()):
                self.combined_statistics[key].append(statistics[key])

            # Log the epoch rewards and successes
            epoch_rewards.append(np.mean(epoch_episode_rewards))
            epoch_success.append(np.mean(epoch_episode_success))

        # Plot the statistics calculated
        if self.plot_stats:
            # Plot the rewards and successes
            rewards_fname = self.output_folder + '/rewards.jpg'
            success_fname = self.output_folder + '/success.jpg'
            plot(epoch_rewards, f_name=rewards_fname, save_fig=True, show_fig=False)
            plot(epoch_success, f_name=success_fname, save_fig=True, show_fig=False)

        # Save the models on the disk
        if self.save_model:
            self.save_model(self.output_folder)

        return self.combined_statistics


if __name__ == '__main__':

    env = retro.make(game='SuperMarioBros-Nes')
    env.reset()
    r = 0
    while True:
        _obs, _rew, done, _info = env.step(env.action_space.sample())
        r += _rew
        if done:
            print(r)
            break
