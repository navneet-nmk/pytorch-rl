"""

This script contains an implementation of the Soft Actor Critic.

This is an off policy Actor critic algorithm with the entropy of
the current policy added to the reward.

The maximization of the augmented reward enables the agent to discover multimodal actions
(Multiple actions that results in reward). It promotes exploration
and is considerably more stable to random seeds variation and hyperparameter
tuning compared to Deep Deterministic Policy Gradient.

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from Memory import Buffer
import Utils.random_process as random_process

USE_CUDA = torch.cuda.is_available()


class SAC(object):

    def __init__(self, state_dim,
                 action_dim,
                 hidden_dim,
                 actor, critic, value_network,
                 target_value_network,
                 polyak_constant,
                 actor_learning_rate,
                 critic_learning_rate,
                 value_learning_rate,
                 num_q_value,
                 num_v_value,
                 batch_size, gamma,
                 random_seed,
                 use_cuda, buffer_capacity
                 ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.q_dim = num_q_value
        self.v_dim = num_v_value
        self.actor = actor
        self.critic = critic
        self.value = value_network
        self.tau = polyak_constant
        self.bs = batch_size
        self.gamma = gamma
        self.seed = random_seed
        self.use_cuda = use_cuda
        self.buffer = Buffer.ReplayBuffer(capacity=buffer_capacity, seed=self.seed)

        self.actor_optim = optim.Adam(lr=actor_learning_rate, params=self.actor.parameters())
        self.critic_optim = optim.Adam(lr=critic_learning_rate, params=self.critic.parameters())
        self.value_optim = optim.Adam(lr=value_learning_rate, params=self.value.parameters())

        self.target_value = target_value_network

        if self.use_cuda:
            self.actor  = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.value = self.value.cuda()
            self.target_value = self.target_value.cuda()

        # Initializing the target networks with the standard network weights
        self.target_value.load_state_dict(self.value.state_dict())

        # Initialize a random exploration noise
        self.random_noise = random_process.OrnsteinUhlenbeckActionNoise(self.action_dim)


# The Policy Network
class StochasticActor(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_dim, use_tanh=False):
        super(StochasticActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.use_tanh = use_tanh

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
        self.hidden_2 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden*2)
        self.output_mu = nn.Linear(in_features=self.hidden*2, out_features=self.action_dim)
        self.output_logvar = nn.Linear(in_features=self.hidden*2, out_features=self.action_dim)

        # Leaky Relu activation function
        self.lrelu = nn.LeakyReLU()

        #Output Activation function
        self.tanh = nn.Tanh()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state):
        x = self.input(state)
        x = self.input(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        x = self.lrelu(x)

        mu = self.output_mu(x)
        logvar = self.output_logvar(x)

        output = self.reparameterize(mu, logvar)

        if self.use_tanh:
            output = self.tanh(output)

        return output, mu, logvar



# Estimates the Q value of a state action pair
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_dim, output_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.output_dim = output_dim

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim+self.action_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
        self.hidden_2 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden*2)
        self.output = nn.Linear(in_features=self.hidden*2, out_features=self.output_dim)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        x = self.lrelu(x)
        output = self.output(x)
        return output


# Estimates the value of a state
class ValueNetwork(nn.Module):

    def __init__(self, state_dim,
                 hidden_dim, output_dim):
        super(ValueNetwork, self).__init__()

        self.state_dim = state_dim
        self.hidden = hidden_dim
        self.output_dim = output_dim

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden * 2)
        self.hidden_2 = nn.Linear(in_features=self.hidden * 2, out_features=self.hidden * 2)
        self.output = nn.Linear(in_features=self.hidden * 2, out_features=self.output_dim)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        x = self.input(state)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        x = self.lrelu(x)
        output = self.output(x)
        return output



