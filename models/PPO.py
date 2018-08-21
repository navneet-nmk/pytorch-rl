"""
Implementation of the Proximal Policy Optimization algorithm

Based on the Trust Region Policy optimization by John Schulman

"""

import torch
import torch.nn as nn
import numpy as np
from Distributions.distributions import init, Categorical
import torch.optim as optim
import torch.nn.functional as F
from Utils.utils import *
from Memory.rollout_storage import RolloutStorage


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):

    def __init__(self,
                 conv_layers,
                 conv_kernel_size,
                 input_channels,
                 height,
                 width,
                 hidden,
                 use_gru,
                 gru_hidden,
                 value,
                 ):
        super(CNNBase, self).__init__()
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.in_channels = input_channels
        self.height = height
        self.width = width
        self.hidden = hidden
        self.use_gru = use_gru
        self.gru_hidden = gru_hidden
        self.value_dim = value

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(self.in_channels, self.conv_layers,self.conv_kernel_size, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(self.conv_layers, self.conv_layers*2, self.conv_kernel_size//2, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(self.conv_layers*2, self.conv_layers, self.conv_kernel_size//2-1, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.conv_layers * self.height//8 * self.width//8, self.hidden)),
            nn.ReLU()
        )

        if self.use_gru:
            self.gru = nn.GRUCell(self.gru_hidden, self.gru_hidden)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(self.hidden, self.value_dim))

    def forward(self, input):
        x = self.main(input)
        value = self.critic_linear(x)

        return value, x


class ActorCritic(nn.Module):
    """
    Implementation of the actor critic architecture in the PPO

    This network outputs the action (Policy Network) and criticizes
    the policy (Critic Network)
    """

    def __init__(self, num_obs,
                 num_actions,
                 action_space,
                 input_channels,
                 num_value, hidden,
                 height, width,
                 use_cnn=True):
        super(ActorCritic, self).__init__()
        self.obs = num_obs
        self.action_dim = num_actions
        self.value = num_value
        self.hidden = hidden
        self.action_space = action_space
        self.use_cnn = use_cnn
        self.in_channels = input_channels
        self.height = height
        self.width = width

        # Common architecture
        self.linear1 = nn.Linear(num_obs, hidden)
        self.hidden1 = nn.Linear(hidden, hidden)

        # Critic Head
        self.value_head = nn.Linear(hidden, self.value)

        # Actor Features
        self.actions = nn.Linear(hidden, self.action_dim)

        # Activation function
        self.activation = nn.Tanh()

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)

        if self.use_cnn:
            self.base = CNNBase(conv_layers=32, conv_kernel_size=8,
                                input_channels=self.in_channels,
                                use_gru=False, gru_hidden=512, hidden=512,
                                height=self.height, width=self.width,
                                value=1)

        else:

            self.base = nn.Sequential(
                self.linear1,
                self.activation,
                self.hidden1,
                self.activation,
            )

        # Old and New Module list for comparison from the previous policy
        # This forms the update step for PPO and helps in the
        # reduction of the variance

        self.module_list_current = [self.linear1, self.hidden1, self.action_mean, self.action_log_std]
        self.module_list_old = [None] * len(self.module_list_current)

    def forward(self, state):
        raise NotImplementedError

    # The function that actually acts
    def act(self, input, deterministic=False):
        value, actor_features = self.base(input)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, dist_entropy

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std, eps=1e-12):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        # print (type(p_mean), type(p_std), type(q_mean), type(q_std))
        # q_mean = Variable(torch.DoubleTensor([q_mean])).expand_as(p_mean)
        # q_std = Variable(torch.DoubleTensor([q_std])).expand_as(p_std)
        numerator = torch.pow((p_mean - q_mean), 2.) + \
            torch.pow(p_std, 2.) - torch.pow(q_std, 2.) #.expand_as(p_std)
        denominator = 2. * torch.pow(q_std, 2.) + eps
        return torch.sum(numerator / denominator + torch.log(q_std) - torch.log(p_std))

    def entropy(self):
        """Gives entropy of current defined prob dist"""
        ent = torch.sum(self.action_log_std + .5 * torch.log(2.0 * np.pi * np.e))
        return ent

    def kl_old_new(self):
        """Gives kld from old params to new params"""
        kl_div = self.kl_div_p_q(self.module_list_old[-2], self.module_list_old[-1], self.action_mean,
                                 self.action_log_std)
        return kl_div

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features, states = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class PPO(object):

    def __init__(self,
                 actor_critic,
                 clip_param,
                 num_epochs,
                 batch_size,
                 value_loss_param,
                 entropy_param,
                 max_grad_norm,
                 learning_rate, use_cuda):
        self.model = actor_critic
        self.num_epochs = num_epochs
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.value_loss_param = value_loss_param
        self.entropy_loss_param = entropy_param
        self.lr= learning_rate
        self.use_cuda = use_cuda
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(lr=self.lr, params=self.model.parameters())

    def train(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # Normalize the advantages
        advantages = (advantages-advantages.mean())/(advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for epoch in range(self.num_epochs):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.batch_size
            )

            for i, sample in enumerate(data_generator):
                observations, actions, \
                rewards, old_action_log_probs, advantage_targets = sample

                values, action_log_probs, dist_entropy = self.model.evaluate_actions(observations, actions)

                ratio = torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advantage_targets
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantage_targets
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(rewards, values)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_param + action_loss -
                 dist_entropy * self.entropy_loss_param).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.num_epochs * self.batch_size

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


# The training class for PPO
class PPOTrainer(object):

    def __init__(self,
                 environment,
                 ppo,
                 forward_dynamics_model,
                 inverse_dynamics_model,
                 forward_dynamics_learning_rate,
                 inverse_dynamics_learning_rate,
                 num_epochs,
                 num_rollouts,
                 num_processes,
                 random_seed,
                 use_cuda, model_save_path):

        self.agent = ppo
        self.fwd_model = forward_dynamics_model
        self.invd_model = inverse_dynamics_model
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.fwd_lr = forward_dynamics_learning_rate
        self.use_cuda = use_cuda
        self.save_path = model_save_path
        self.inv_lr = inverse_dynamics_learning_rate
        self.env = environment
        self.num_processes = num_processes
        self.random_seed = random_seed

        # Environment Details
        self.action_space = self.env.action_space
        self.action_shape = self.env.action_space.n
        self.obs_shape = self.env.observation_space.n

        # Create the rollout storage
        self.rollout_storage = RolloutStorage(num_steps=self.num_rollouts,
                                              action_shape=self.action_shape,
                                              action_space=self.action_space,
                                              num_processes=self.num_processes,
                                              obs_shape=self.obs_shape,
                                              use_cuda=self.use_cuda)

        # Define the optimizers for the forward dynamics and inverse dynamics models
        self.fwd_optim = optim.Adam(lr=self.fwd_lr, params=self.fwd_model.parameters())
        self.inverse_optim = optim.Adam(lr=self.inv_lr, params=self.invd_model.parameters())

    # Calculation of the curiosity reward
    def calculate_intrinsic_reward(self, obs, action, new_obs):
        # Encode the obs and new_obs
        obs_encoding = self.invd_model.encode(obs)
        new_obs_encoding = self.invd_model.encode(new_obs)

        # Pass the action and obs encoding to forward dynamic model
        pred_new_obs_encoding = self.fwd_model(obs_encoding, action)
        reward = F.mse_loss(pred_new_obs_encoding, new_obs_encoding)

        return reward

    # Rollout Collection function
    def collect_rollouts(self):
        observation = self.env.reset()
        observation = to_tensor(observation, use_cuda=self.use_cuda)
        for r in range(self.num_rollouts):
            value, action, action_log_prob, dist_entropy = self.agent.act(observation)
            next_observation, reward, done, _ = self.env.step(action)
            intrinsic_reward = self.calculate_intrinsic_reward(obs=observation,
                                                               action=action,
                                                               new_obs=next_observation)
            # Store in the rollout storage
            self.rollout_storage.insert(step=r,
                                        current_obs=observation,
                                        action=action,
                                        action_log_prob=action_log_prob,
                                        intrinsic_reward=intrinsic_reward,
                                        reward=reward,
                                        value_pred=value)

            if done:
                break

    def train(self):
        # Update the agent
        self.agent.train(self.rollout_storage)
        # Update the inverse dynamics model

        # Update the forward dynamics model
        self.fwd_model.train()

        # Update the inverse dynamics model
        self.invd_model.train()














