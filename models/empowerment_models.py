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
from tensorboardX import SummaryWriter
import torch.nn.functional as F

def epsilon_greedy_exploration():
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 300
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    return epsilon_by_frame


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
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        state = torch.unsqueeze(state, dim=0)
        x = self.conv1(state)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view((-1, self.height//4*self.width//4*self.conv_layers))
        x = self.hidden_1(x)
        x = self.relu(x)
        encoded_state = self.output(x)
        encoded_state = self.tanh_activ(encoded_state)
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
                 use_encoding=True):
        super(forward_dynamics_model, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.use_encoding = use_encoding

        # Forward Dynamics Model Architecture

        # Given the current state and the action, this network predicts the next state

        self.layer1 = nn.Linear(in_features=self.state_space, out_features=self.hidden)
        self.layer2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.layer3 = nn.Linear(in_features=self.hidden+self.action_space, out_features=self.hidden)
        self.output = nn.Linear(in_features=self.hidden, out_features=self.state_space)

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
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
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

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output.weight)

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
                 target_forward_dynamics,
                 statistics_network,
                 target_stats_network,
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
                 size_dqn_replay_buffer,
                 random_seed,
                 polyak_constant,
                 discount_factor,
                 batch_size,
                 action_space,
                 model_output_folder,
                 print_every=2000,
                 update_network_every=2000,
                 plot_every=5000,
                 intrinsic_param=0.05,
                 use_mine_formulation=True,
                 use_cuda=False,
                 save_models=True,
                 plot_stats=False,
                 verbose=True):

        self.encoder = encoder
        self.fwd = forward_dynamics
        self.target_fwd = target_forward_dynamics
        self.stats = statistics_network
        self.target_stats = target_stats_network
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

        self.fwd_limit = fwd_dynamics_limit
        self.stats_limit = stats_network_limit
        self.policy_limit = policy_limit

        self.print_every = print_every
        self.update_every = update_network_every
        self.plot_every = plot_every

        self.statistics = defaultdict(float)
        self.combined_statistics = defaultdict(list)

        # Tensorboard writer
        self.writer = SummaryWriter()

        # Fix the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.replay_buffer = Buffer.ReplayBuffer(capacity=size_replay_buffer,
                                                 seed=self.random_seed)

        self.dqn_replay_buffer = Buffer.ReplayBuffer(capacity=size_dqn_replay_buffer,
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
    def store_transition(self, buffer, state, new_state, action, reward, done):
        buffer.push(state, action, new_state, reward, done)

    # Update the networks using polyak averaging
    def update_networks(self, hard_update=True):
        if hard_update:
            for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_(param.data)
            #for target_param, param in zip(self.target_stats.parameters(), self.stats.parameters()):
            #    target_param.data.copy_(param.data)
            #for target_param, param in zip(self.target_fwd.parameters(), self.fwd.parameters()):
            #    target_param.data.copy_(param.data)
        else:
            for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
                target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    # Train the policy network
    def train_policy(self, clip_gradients=True):
        # Sample mini-batch from the replay buffer uniformly or from the prioritized experience replay.

        # If the size of the buffer is less than batch size then return
        if self.dqn_replay_buffer.get_buffer_size() < self.batch_size:
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        q_values = self.policy_network(states)
        next_q_values = self.policy_network(new_states)

        next_q_state_values = self.target_policy_network(new_states).detach()

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        expected_q_value = expected_q_value.detach()
        td_loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.policy_optim.zero_grad()
        td_loss.backward()
        if clip_gradients:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.policy_optim.step()

        return td_loss

    def train_forward_dynamics(self, clamp_gradients=False, use_difference_representation=True):

        if self.replay_buffer.get_buffer_size() < self.batch_size:
            return None

        transitions = self.replay_buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        states = Variable(torch.cat(states))
        new_states = Variable(torch.cat(new_states))
        actions = Variable(torch.cat(actions))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            new_states = new_states.cuda()

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
        if clamp_gradients:
            for param in self.fwd.parameters():
                param.grad.data.clamp_(-1, 1)
        self.fwd_optim.step()

        return mse_error

    def train_statistics_network(self, use_jenson_shannon_divergence=True,
                                 use_target_forward_dynamics=False,
                                 use_target_stats_network=False,
                                 clamp_gradients=False):

        if self.replay_buffer.get_buffer_size() < self.batch_size:
            return None, None, None

        transitions = self.replay_buffer.sample_batch(self.batch_size)
        batch = Buffer.Transition(*zip(*transitions))

        # Get the separate values from the named tuple
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

        all_actions = self.get_all_actions(self.action_space)
        all_actions = Variable(torch.cat(all_actions))

        new_state_marginals = []
        for state in states:
            state = state.expand(self.action_space, -1)
            if use_target_forward_dynamics:
                n_s = self.target_fwd(state, all_actions)
            else:
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

        p_s_ta = self.target_stats(new_states, actions)
        p_s_t_a = self.target_stats(new_state_marginals, actions)

        if use_jenson_shannon_divergence:
            # Improves stability and gradients are unbiased
            if use_target_stats_network:
                mutual_information = -F.softplus(-p_s_ta) - F.softplus(p_s_t_a)
            else:
                mutual_information = -F.softplus(-p_sa) - F.softplus(p_s_a)
            lower_bound = torch.mean(-F.softplus(-p_sa)) - torch.mean(F.softplus(p_s_a))
        else:
            # Use KL Divergence
            if use_target_stats_network:
                mutual_information = p_s_ta - torch.log(torch.exp(p_s_t_a))
            else:
                mutual_information = p_sa - torch.log(torch.exp(p_s_a))
            lower_bound = torch.mean(p_sa) - torch.log(torch.mean(torch.exp(p_s_a)))

        # Maximize the mutual information
        loss = -lower_bound
        self.stats_optim.zero_grad()
        loss.backward()
        # Clamp the gradients
        if clamp_gradients:
            for param in self.stats.parameters():
                param.grad.data.clamp_(-1, 1)
        self.stats_optim.step()

        # Store in the dqn replay buffer

        mutual_information = torch.squeeze(mutual_information, dim=-1)
        mutual_information = mutual_information.detach()

        rewards_combined = rewards + self.intrinsic_param*mutual_information
        # Store the updated reward transition in the replay buffer
        self.store_transition(state=states,
                              action=actions,
                              new_state=new_states,
                              reward=rewards_combined,
                              done=dones, buffer=self.dqn_replay_buffer)

        return loss, rewards, mutual_information, lower_bound

    def plot(self, frame_idx, rewards, losses, placeholder_name, output_folder):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards)))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        file_name_pre = output_folder+placeholder_name
        fig.savefig(file_name_pre+str(frame_idx)+'.jpg')
        plt.close(fig)

    def save_m(self):
        print("Saving the models")
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

    def train(self):
        # Starting time
        start_time = time.time()

        # Initialize the statistics dictionary
        statistics = self.statistics

        episode_rewards_history = deque(maxlen=100)
        episode_success_history = deque(maxlen=100)

        epoch_episode_rewards = []
        epoch_episode_success = []
        epoch_episode_steps = []

        # Epoch Rewards and success
        epoch_rewards = []
        epoch_success = []

        # Initialize the training with an initial state
        state = self.env.reset()

        # Initialize the losses
        policy_losses = [0]
        fwd_model_loss = []
        stats_model_loss = [0]
        episode_reward = 0
        intrinsic_reward = []
        episode_success = 0
        episode_step = 0
        epoch_actions = []

        # Check whether to use cuda or not
        state = to_tensor(state, use_cuda=self.use_cuda)
        state = self.encoder(state)

        for frame_idx in range(1, self.num_frames+1):
            epsilon_by_frame = epsilon_greedy_exploration()
            epsilon = epsilon_by_frame(frame_idx)
            action = self.policy_network.act(state, epsilon)

            # Execute the action
            next_state, reward, done, success = self.env.step(action.item())
            episode_reward += reward

            next_state = to_tensor(next_state, use_cuda=self.use_cuda)
            next_state = self.encoder(next_state)
            reward = torch.tensor([reward], dtype=torch.float)

            done_bool = done * 1
            done_bool = torch.tensor([done_bool], dtype=torch.float)
            self.store_transition(state=state, new_state=next_state,
                                  action=action, done=done_bool,reward=reward, buffer=self.replay_buffer)

            state = next_state

            if done:
                epoch_episode_rewards.append(episode_reward)
                episode_rewards_history.append(episode_reward)
                episode_success_history.append(episode_success)
                epoch_episode_success.append(episode_success)
                epoch_episode_steps.append(episode_step)
                # Add episode reward to tensorboard
                self.writer.add_scalar('Extrinsic Reward', episode_reward, frame_idx)
                episode_reward = 0
                episode_step = 0
                episode_success = 0
                state = self.env.reset()
                state = to_tensor(state, use_cuda=self.use_cuda)
                state = self.encoder(state)

            # Train the forward dynamics model
            if len(self.replay_buffer) > self.fwd_limit:
                fwd_loss = 0
                if frame_idx % 1 ==0:
                    for t in range(self.num_fwd_train_steps):
                        mse_loss = self.train_forward_dynamics()
                        fwd_loss += mse_loss
                        fwd_model_loss.append(mse_loss.data[0])
                    self.writer.add_scalar('Forward Dynamics Loss', fwd_loss/self.num_fwd_train_steps, frame_idx)


            # Train the statistics network
            if len(self.replay_buffer) > self.stats_limit:
                # This will also append the updated transitions to the replay buffer
                stat_loss = 0
                if frame_idx%1 == 0:
                    for s in range(self.num_stats_train_steps):
                        stats_loss, extrinsic_rewards, intrinsic_rewards, lower_bound = self.train_statistics_network()
                        stat_loss += stats_loss
                        intrinsic_reward.append(torch.mean(intrinsic_rewards))
                        stats_model_loss.append(stats_loss.data[0])
                    # Add to tensorboard
                    self.writer.add_scalar('stats_loss', stat_loss/self.num_stats_train_steps, frame_idx)

            # Train the policy
            if len(self.dqn_replay_buffer) > self.policy_limit:
                p_loss = 0
                if frame_idx % 1 == 0:
                    for m in range(self.train_epochs):
                        policy_loss = self.train_policy()
                        p_loss += policy_loss
                        policy_losses.append(policy_loss.data[0])
                    self.writer.add_scalar('policy_loss', p_loss/self.train_epochs, frame_idx)

            # Log stats
            duration = time.time() - start_time
            statistics['rollout/rewards'] = np.mean(epoch_episode_rewards)
            statistics['rollout/rewards_history'] = np.mean(episode_rewards_history)
            statistics['rollout/successes'] = np.mean(epoch_episode_success)
            statistics['rollout/successes_history'] = np.mean(episode_success_history)
            statistics['rollout/actions_mean'] = np.mean(epoch_actions)
            statistics['total/duration'] = duration

            self.writer.add_scalar('Length of Buffer', len(self.replay_buffer), frame_idx)
            self.writer.add_scalar('Mean Reward', np.mean(epoch_episode_rewards), frame_idx)
            self.writer.add_scalar('Length of DQN Buffer', len(self.dqn_replay_buffer), frame_idx)

            # Print the statistics
            if self.verbose:
                if frame_idx % self.print_every == 0:
                    print('Forward Dynamics Loss ', str(np.mean(fwd_model_loss)))
                    print('Statistics Network Loss ', str(np.mean(stats_model_loss)))
                    print('Policy Loss ', str(np.mean(policy_losses)))
                    print('Mean Reward ', str(np.mean(epoch_episode_rewards)))

            if self.plot_stats:
                if frame_idx % self.plot_every == 0:
                # Plot the statistics calculated
                    self.plot(frame_idx=frame_idx, rewards=epoch_episode_rewards, losses=policy_losses,
                            output_folder=self.output_folder, placeholder_name='/DQN_montezuma_intrinsic')


            # Update the target network
            if frame_idx % self.update_every:
                self.update_networks()

            # Log the combined statistics for all epochs
            for key in sorted(statistics.keys()):
                self.combined_statistics[key].append(statistics[key])

            # Log the epoch rewards and successes
            epoch_rewards.append(np.mean(epoch_episode_rewards))
            epoch_success.append(np.mean(epoch_episode_success))

        # Close the tensorboard writer
        self.writer.close()
        # Save the models
        self.save_m()
        return self.combined_statistics


if __name__ == '__main__':

    # Setup the environment
    env = gym.make('MontezumaRevenge-v4')
    # Add the required environment wrappers
    env = env_wrappers.warp_wrap(env, height=84, width=84)
    env = env_wrappers.wrap_pytorch(env)

    action_space = env.action_space.n
    state_space = env.observation_space
    height = 84
    width = 84
    num_hidden_units = 64

    encoder = Encoder(state_space=num_hidden_units, conv_kernel_size=3, conv_layers=16,
                      hidden=64, input_channels=1, height=height,
                      width=width)
    policy_model = QNetwork(env=env, state_space=num_hidden_units,
                             action_space=action_space, hidden=num_hidden_units)
    target_policy_model = QNetwork(env=env, state_space=num_hidden_units,
                             action_space=action_space, hidden=num_hidden_units)
    stats_network = StatisticsNetwork(action_space=action_space, state_space=num_hidden_units,
                                      hidden=64, output_dim=1)
    target_stats_network = StatisticsNetwork(action_space=action_space, state_space=num_hidden_units,
                                      hidden=64, output_dim=1)
    forward_dynamics_network = forward_dynamics_model(action_space=action_space, hidden=64,
                                                      state_space=num_hidden_units)
    target_forward_dynamics_network = forward_dynamics_model(action_space=action_space, hidden=64,
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
        stats_lr=1e-3,
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
        size_replay_buffer=100000,
        size_dqn_replay_buffer=100000,
        plot_stats=True,
        print_every=2000,
        plot_every=20000,
        intrinsic_param=0.2,
        target_stats_network=target_stats_network,
        target_forward_dynamics=target_forward_dynamics_network
    )

    # Train
    empowerment_model.train()

