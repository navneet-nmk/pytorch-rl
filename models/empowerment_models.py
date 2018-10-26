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
import Environments.env_wrappers as env_wrappers
import random
import math
#from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.multiprocessing as mp
torch.backends.cudnn.enabled = False

def epsilon_greedy_exploration():
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 50000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    return epsilon_by_frame

# L2 normalize the vector
def l2_normalize(tensor):
    l2_norm = torch.sqrt(torch.sum(torch.pow(tensor, 2)))
    return tensor/l2_norm

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

        # Use batchnormalization in encoder to stabilize the training
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.bn2 = nn.BatchNorm2d(self.conv_layers)

        # Relu activation
        self.relu = nn.ReLU(inplace=True)

        # Hidden Layers
        self.hidden_1 = nn.Linear(in_features=self.height // 4 * self.width // 4 * self.conv_layers,
                                  out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.state_space)
        self.tanh_activ = nn.Tanh()

        # Initialize the weights of the network (Since this is a random encoder, these weights will
        # remain static during the training of other networks).
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        #nn.init.xavier_uniform_(self.bn1.weight)
        #nn.init.xavier_uniform_(self.bn2.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        x = self.conv1(state)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view((-1, self.height//4*self.width//4*self.conv_layers))
        x = self.hidden_1(x)
        x = self.relu(x)
        encoded_state = self.output(x)
        #encoded_state = self.tanh_activ(encoded_state)
        # L2 Normalize the output of the encoder
        #encoded_state = l2_normalize(encoded_state)
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

        # Relu activation
        self.relu = nn.ReLU(inplace=True)

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
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.relu(x)
            x = self.layer4(x)
            x = self.relu(x)
            x = self.hidden_1(x)
        else:
            x = self.conv1(state)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = x.view((-1, self.height // 16 * self.width // 16 * self.conv_layers * 2))
            x = self.hidden_1(x)

        x = self.relu(x)
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

    def __init__(self,
                 state_space,
                 action_space,
                 hidden,
                 use_encoding=True,
                 return_gaussians=False):
        super(forward_dynamics_model, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.use_encoding = use_encoding
        self.return_gaussians = return_gaussians

        # Forward Dynamics Model Architecture

        # Given the current state and the action, this network predicts the next state

        self.layer1 = nn.Linear(in_features=self.state_space, out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
        self.layer3 = nn.Linear(in_features=self.hidden*2+self.action_space, out_features=self.hidden*2)
        if self.return_gaussians:
            self.output_mu = nn.Linear(in_features=self.hidden, out_features=self.state_space)
            self.output_logvar = nn.Linear(in_features=self.hidden, out_features=self.state_space)
        self.output = nn.Linear(in_features=self.hidden*2, out_features=self.state_space)

        # Relu activation
        self.relu = nn.ReLU(inplace=True)

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def one_hot_action(self, batch_size, action):
        ac = torch.zeros(batch_size, self.action_space)
        for i in range(batch_size):
            ac[i, action[i]] = 1
        return ac

    def forward(self, current_state, action):
        bs, _ = current_state.shape
        x = self.layer1(current_state)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        action = action.unsqueeze(1)
        ac = self.one_hot_action(batch_size=bs, action=action)
        x = torch.cat([x, ac], dim=-1)
        x = self.layer3(x)
        x = self.relu(x)
        if self.return_gaussians:
            output_mu, output_logvar = self.output_mu(x), self.output_logvar(x)
            return output_mu, output_logvar
        output = self.output(x)
        return output


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
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.action_space)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state), requires_grad=False)
            q_value = self.forward(state)
            # Action corresponding to the max Q Value for the state action pairs

            action = q_value.max(1)[1].view(1)
        else:
            action = torch.tensor([random.randrange(self.env.action_space.n)], dtype=torch.long)

        return action

# Convolutional Policy Network
class ConvolutionalQNetwork(nn.Module):
    def __init__(self, env, state_height,
                 state_width,
                 action_space,
                 input_channels,
                 hidden):
        super(ConvolutionalQNetwork, self).__init__()

        self.height = state_height
        self.width = state_width
        self.hidden = hidden
        self.env = env
        self.action_space = action_space
        self.in_channels = input_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden, kernel_size=self.conv_kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=self.conv_kernel_size, stride=2, padding=1),
            nn.ReLU(),
        )

        self.layers = nn.Sequential(
            nn.Linear(self.height//4*self.width//4*self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.action_space)
        )

        # Separate head for intrinsic rewards
        self.intrinsic_layers = nn.Sequential(
            nn.Linear(self.height // 4 * self.width // 4 * self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.action_space)
        )

    def forward(self, state):
        conv = self.conv_layers(state)
        # Return the Q values for all the actions
        output = self.layers(conv)
        return output

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state), requires_grad=False)
            q_value = self.forward(state)
            # Action corresponding to the max Q Value for the state action pairs

            action = q_value.max(1)[1].view(1)
        else:
            action = torch.tensor([random.randrange(self.env.action_space.n)], dtype=torch.long)

        return action



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

        # Relu activation
        self.relu = nn.ReLU(inplace=True)

    def one_hot_action(self, batch_size, action):
        ac = torch.zeros(batch_size, self.action_space)
        for i in range(batch_size):
            ac[i, action[i]] = 1
        return ac

    def forward(self, next_state, action):
        bs, _ = next_state.shape
        action = action.unsqueeze(1)
        ac = self.one_hot_action(batch_size=bs, action=action)
        s = torch.cat([next_state, ac], dim=-1)
        x = self.layer1(s)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        output = self.output(x)
        return output


class EmpowermentTrainer(object):

    def __init__(self,
                 env,
                 encoder,
                 forward_dynamics,
                 statistics_network,
                 target_policy_network,
                 policy_network,
                 forward_dynamics_lr,
                 stats_lr,
                 policy_lr,
                 num_train_epochs,
                 num_frames,
                 num_fwd_train_steps,
                 num_stats_train_steps,
                 fwd_dynamics_limit,
                 stats_network_limit,
                 policy_limit,
                 size_replay_buffer,
                 random_seed,
                 polyak_constant,
                 discount_factor,
                 batch_size,
                 action_space,
                 model_output_folder,
                 save_epoch,
                 target_stats_network=None,
                 target_fwd_dynamics_network=None,
                 clip_rewards=True,
                 clip_augmented_rewards=False,
                 print_every=2000,
                 update_network_every=2000,
                 plot_every=5000,
                 intrinsic_param=0.01,
                 non_episodic_intrinsic=True,
                 use_mine_formulation=True,
                 use_cuda=False,
                 save_models=True,
                 plot_stats=False,
                 verbose=True):

        self.encoder = encoder
        self.fwd = forward_dynamics
        self.stats = statistics_network
        self.use_cuda = use_cuda
        self.policy_network = policy_network
        self.target_policy_network = target_policy_network
        self.output_folder = model_output_folder
        self.use_mine_formulation = use_mine_formulation
        self.env = env
        self.train_epochs = num_train_epochs
        self.num_frames = num_frames
        self.num_fwd_train_steps = num_fwd_train_steps
        self.num_stats_train_steps = num_stats_train_steps
        self.fwd_lr = forward_dynamics_lr
        self.stats_lr = stats_lr
        self.policy_lr = policy_lr
        self.random_seed = random_seed
        self.save_models = save_models
        self.plot_stats = plot_stats
        self.verbose = verbose
        self.intrinsic_param = intrinsic_param
        self.save_epoch = save_epoch
        self.clip_rewards = clip_rewards
        self.clip_augmented_rewards = clip_augmented_rewards
        self.max = torch.zeros(1)
        self.min = torch.zeros(1)

        self.fwd_limit = fwd_dynamics_limit
        self.stats_limit = stats_network_limit
        self.policy_limit = policy_limit

        self.print_every = print_every
        self.update_every = update_network_every
        self.plot_every = plot_every
        self.non_episodic = non_episodic_intrinsic

        self.statistics = defaultdict(float)
        self.combined_statistics = defaultdict(list)

        self.target_stats_network = target_stats_network
        self.target_fwd_dynamics_network = target_fwd_dynamics_network

        # Fix the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.replay_buffer = Buffer.ReplayBuffer(capacity=size_replay_buffer,
                                                 seed=self.random_seed)

        self.tau = polyak_constant
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.action_space = action_space

        torch.manual_seed(self.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.random_seed)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.invd = self.invd.cuda()
            self.fwd = self.fwd.cuda()
            self.policy_network = self.policy_network.cuda()
            self.source_distribution = self.source_distribution.cuda()

        self.fwd_optim = optim.Adam(params=self.fwd.parameters(), lr=self.fwd_lr)
        self.policy_optim = optim.Adam(params=self.policy_network.parameters(), lr=self.policy_lr)
        self.stats_optim = optim.Adam(params=self.stats.parameters(), lr=self.stats_lr)
        # Update the policy and target policy networks
        self.update_networks()

    def get_all_actions(self, action_space):
        all_actions = []
        for i in range(action_space):
            all_actions.append(torch.LongTensor([i]))
        return all_actions

    def softplus(self, z):
        return torch.log(1+ torch.exp(z))

    # Store the transition into the replay buffer
    def store_transition(self, state, new_state, action, reward, done):
        self.replay_buffer.push(state, action, new_state, reward, done)

    # Update the networks using polyak averaging
    def update_networks(self, hard_update=True):
        if hard_update:
            for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_stats_network.parameters(), self.stats.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.target_fwd_dynamics_network.parameters(), self.fwd.parameters()):
                target_param.data.copy_(param.data)
        else:
            for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    # Train the policy network
    def train_policy(self, batch, rewards, clip_gradients=True):

        states = batch['states']
        new_states = batch['new_states']
        actions = batch['actions']
        dones = batch['dones']

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        q_values = self.policy_network(states)
        next_q_values = self.policy_network(new_states)
        with torch.no_grad():
            next_q_state_values = self.target_policy_network(new_states).detach()

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        expected_q_value = expected_q_value.detach()
        # Use smooth l1 loss with caution. refer to https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
        td_loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.policy_optim.zero_grad()
        td_loss.backward()
        if clip_gradients:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.policy_optim.step()

        return td_loss

    def train_forward_dynamics(self, batch, clamp_gradients=False, use_difference_representation=True):

        states = batch['states']
        new_states = batch['new_states']
        actions = batch['actions']

        if use_difference_representation:
            # Under this representation, the model predicts the difference between the current state and the next state.
            diff_new_states  = self.fwd(states, actions)
            predicted_new_states = states + diff_new_states
        else:
            predicted_new_states = self.fwd(states, actions)

        mse_error = F.mse_loss(predicted_new_states, new_states)
        self.fwd_optim.zero_grad()
        mse_error.backward()
        # Clamp the gradients
        self.fwd_optim.step()

        return mse_error

    def update_intrinsic_param(self, param, rewards):
        t = torch.max(torch.FloatTensor([1]), torch.mean(rewards) )
        new_param_val = param/t
        return new_param_val

    def train_statistics_network(self, batch, use_jenson_shannon_divergence=True):

        states = batch['states']
        new_states = batch['new_states']
        actions = batch['actions']
        rewards = batch['rewards']

        all_actions = self.get_all_actions(self.action_space)
        all_actions = Variable(torch.cat(all_actions))

        new_state_marginals = []
        for state in states:
            state = state.expand(self.action_space, -1)
            with torch.no_grad():
                n_s = self.fwd(state, all_actions)
            n_s = n_s.detach()
            n_s = n_s + state
            n_s = torch.mean(n_s, dim=0)
            n_s = torch.unsqueeze(n_s, dim=0)
            new_state_marginals.append(n_s)

        new_state_marginals = tuple(new_state_marginals)
        new_state_marginals = Variable(torch.cat(new_state_marginals), requires_grad=False)

        p_sa = self.stats(new_states, actions)
        p_s_a = self.stats(new_state_marginals, actions)

        if use_jenson_shannon_divergence:
            # Improves stability and gradients are unbiased
            # But use the kl divergence representation for the reward
            mutual_information =-F.softplus(-p_sa) - F.softplus(p_s_a)
            lower_bound = torch.mean(-F.softplus(-p_sa)) - torch.mean(F.softplus(p_s_a))
        else:
            # Use KL Divergence
            mutual_information = -F.softplus(-p_sa) - F.softplus(p_s_a)

        # Maximize the mutual information
        loss = -lower_bound
        self.stats_optim.zero_grad()
        loss.backward()
        self.stats_optim.step()


        mutual_information = mutual_information.squeeze(-1)
        mutual_information = mutual_information.detach()

        mutual_information = torch.clamp(input=mutual_information, min=-1., max=1.)

        augmented_rewards = rewards + self.intrinsic_param*mutual_information
        augmented_rewards.detach()


        return loss, augmented_rewards

    def plot(self, frame_idx, rewards, placeholder_name, output_folder, mean_rewards):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards)))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards)))
        plt.plot(mean_rewards)
        file_name_pre = output_folder+placeholder_name
        fig.savefig(file_name_pre+str(frame_idx)+'.jpg')
        plt.close(fig)

    def save_rewards(self, ep_rewards, mean_rewards):
        np.save(file='epoch_rewards.npy', arr=ep_rewards)
        np.save(file='mean_rewards.npy', arr=mean_rewards)


    def save_m(self):
        torch.save(
            self.encoder.state_dict(),
            '{}/encoder.pt'.format(self.output_folder)
        )
        torch.save(
            self.stats.state_dict(),
            '{}/statistics_network.pt'.format(self.output_folder)
        )
        torch.save(
            self.policy_network.state_dict(),
            '{}/policy_network.pt'.format(self.output_folder)
        )
        torch.save(
            self.fwd.state_dict(),
            '{}/forward_dynamics_network.pt'.format(self.output_folder)
        )

    def get_train_variables(self, batch):

        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states), requires_grad=False)
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        b = defaultdict()
        b['states'] = states
        b['actions'] = actions
        b['rewards'] = rewards
        b['new_states'] = new_states
        b['dones'] = dones

        return b

    def normalize(self, r):
        normalized = (r - torch.mean(r))/(torch.std(r) + 1e-10)
        return normalized

    def train(self):
        epoch_episode_rewards = []

        # Initialize the training with an initial state
        state = self.env.reset()

        # Initialize the losses
        episode_reward = 0
        # Check whether to use cuda or not
        state = to_tensor(state, use_cuda=self.use_cuda)

        fwd_loss = 0
        stats_loss = 0
        policy_loss = 0


        # Mean rewards
        mean_rewards = []
        with torch.no_grad():
            state = self.encoder(state)
        state = state.detach()

        for frame_idx in range(1, self.num_frames+1):
            epsilon_by_frame = epsilon_greedy_exploration()
            epsilon = epsilon_by_frame(frame_idx)
            action = self.policy_network.act(state, epsilon)

            # Execute the action
            next_state, reward, done, success = self.env.step(action.item())
            episode_reward += reward

            reward = np.sign(reward)

            next_state = to_tensor(next_state, use_cuda=self.use_cuda)
            with torch.no_grad():
                next_state = self.encoder(next_state)

            next_state = next_state.detach()

            reward = torch.tensor([reward], dtype=torch.float)

            done_bool = done * 1
            done_bool = torch.tensor([done_bool], dtype=torch.float)

            # Store in the replay buffer
            self.store_transition(state=state, new_state=next_state,
                                  action=action, done=done_bool,reward=reward)

            state = next_state

            if done:
                epoch_episode_rewards.append(episode_reward)
                # Add episode reward to tensorboard
                episode_reward = 0
                state = self.env.reset()
                state = to_tensor(state, use_cuda=self.use_cuda)
                state = self.encoder(state)

            # Train the forward dynamics model
            if len(self.replay_buffer) > self.fwd_limit:
                # Sample a minibatch from the replay buffer
                transitions = self.replay_buffer.sample_batch(self.batch_size)
                batch = Buffer.Transition(*zip(*transitions))
                batch = self.get_train_variables(batch)
                mse_loss = self.train_forward_dynamics(batch=batch)
                fwd_loss += mse_loss.item()
                if frame_idx % self.print_every == 0:
                    print('Forward Dynamics Loss :', fwd_loss/(frame_idx-self.fwd_limit))

            # Train the statistics network and the policy
            if len(self.replay_buffer) > self.policy_limit:
                transitions = self.replay_buffer.sample_batch(self.batch_size)
                batch = Buffer.Transition(*zip(*transitions))
                batch = self.get_train_variables(batch)
                loss, aug_rewards = self.train_statistics_network(batch=batch)

                p_loss = self.train_policy(batch=batch, rewards=aug_rewards)

                stats_loss += loss.item()
                policy_loss += p_loss.item()

                if frame_idx % self.print_every == 0:
                    print('Statistics Loss: ', stats_loss/(frame_idx-self.policy_limit))
                    print('Policy Loss: ', policy_loss/(frame_idx - self.policy_limit))


            # Print the statistics
            if self.verbose:
                if frame_idx % self.print_every == 0:
                    print('Mean Reward ', str(np.mean(epoch_episode_rewards)))
                    print('Sum of Rewards ', str(np.sum(epoch_episode_rewards)))
                    mean_rewards.append(np.mean(epoch_episode_rewards))

            if self.plot_stats:
                if frame_idx % self.plot_every == 0:
                # Plot the statistics calculated
                    self.plot(frame_idx=frame_idx, rewards=epoch_episode_rewards,
                              mean_rewards=mean_rewards,
                            output_folder=self.output_folder, placeholder_name='/DQN_montezuma_intrinsic')

            # Update the target network
            if frame_idx % self.update_every == 0:
                self.update_networks()

            # Save the models and the rewards file
            if frame_idx % self.save_epoch == 0:
                self.save_m()
                self.save_rewards(ep_rewards=epoch_episode_rewards, mean_rewards=mean_rewards)

        self.save_m()

if __name__ == '__main__':

    # Setup the environment
    # Frame skipping is added in the wrapper
    env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # Add the required environment wrappers
    env = env_wrappers.warp_wrap(env, height=84, width=84)
    env = env_wrappers.wrap_pytorch(env)

    action_space = env.action_space.n
    state_space = env.observation_space
    height = 84
    width = 84
    num_hidden_units = 128
    # The input to the encoder is the stack of the last 4 frames of the environment.
    encoder = Encoder(state_space=num_hidden_units, conv_kernel_size=3, conv_layers=32,
                      hidden=128, input_channels=4, height=height,
                      width=width)
    policy_model = QNetwork(env=env, state_space=num_hidden_units,
                             action_space=action_space, hidden=num_hidden_units)
    target_policy_model = QNetwork(env=env, state_space=num_hidden_units,
                             action_space=action_space, hidden=num_hidden_units)
    stats_network = StatisticsNetwork(action_space=action_space, state_space=num_hidden_units,
                                      hidden=128, output_dim=1)
    forward_dynamics_network = forward_dynamics_model(action_space=action_space, hidden=128,
                                                      state_space=num_hidden_units)

    # Defining targets networks to possibily improve the stability of the algorithm
    target_stats_network = StatisticsNetwork(action_space=action_space, state_space=num_hidden_units,
                                             hidden=128, output_dim=1)
    target_fwd_dynamics_network = forward_dynamics_model(action_space=action_space, hidden=128,
                                                         state_space=num_hidden_units)

    # Define the model
    empowerment_model = EmpowermentTrainer(
        action_space=action_space,
        batch_size=32,
        discount_factor=0.99,
        encoder=encoder,
        statistics_network=stats_network,
        forward_dynamics=forward_dynamics_network,
        policy_network=policy_model,
        target_policy_network=target_policy_model,
        env=env,
        forward_dynamics_lr=1e-3,
        stats_lr=1e-4,
        policy_lr=1e-4,
        fwd_dynamics_limit=1000,
        stats_network_limit=5000,
        model_output_folder='montezuma_dqn',
        num_frames=100000000,
        num_fwd_train_steps=1,
        num_stats_train_steps=1,
        num_train_epochs=1,
        policy_limit=10000,
        polyak_constant=0.99,
        random_seed=2450,
        size_replay_buffer=1000000,
        plot_stats=True,
        print_every=4000,
        plot_every=100000,
        intrinsic_param=0.025,
        save_epoch=20000,
        update_network_every=2000,
        target_stats_network=target_stats_network,
        target_fwd_dynamics_network=target_fwd_dynamics_network
    )

    # Train
    empowerment_model.train()

