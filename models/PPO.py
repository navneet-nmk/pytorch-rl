"""
Implementation of the Proximal Policy Optimization algorithm

Based on the Trust Region Policy optimization by John Schulman

"""

import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """
    Implementation of the actor critic architecture in the PPO

    This network outputs the action (Policy Network) and criticizes
    the policy (Critic Network)
    """

    def __init__(self, num_obs, num_actions, num_value, hidden):
        super(ActorCritic, self).__init__()
        self.obs = num_obs
        self.action = num_actions
        self.value = num_value
        self.hidden = hidden

        # Actor architecture
        self.linear1 = nn.Linear(num_obs, hidden)
        self.hidden1 = nn.Linear(hidden, hidden)

        self.actions_mean = nn.Linear(hidden, self.action)
        self.actions_mean.weight.data.mul_(0.1)
        self.actions_mean.bias.data.mul_(0.0)
        self.actions_log_std = nn.Parameter(torch.zeros(1, self.action))

        # Critic Head
        self.value_head = nn.Linear(hidden, self.value)

        # Activation function
        self.activation = nn.Tanh()

        # Old and New Module list for comparison from the previous policy
        # This forms the update step for PPO and helps in the
        # reduction of the variance

        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std]
        self.module_list_old = [None] * len(self.module_list_current)


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.hidden1(x)
        x = self.activation(x)

        action_mean = self.actions_mean(x)
        actions_log_std = self.actions_log_std(x).expand_as(action_mean)
        action_std =  torch.exp(actions_log_std)

        value = self.value_head(x)

        return action_mean, actions_log_std, action_std, value

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std, eps):
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