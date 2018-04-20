import matplotlib.pyplot as plt
import numpy as np
import math
import gym
import torch
from torch.autograd import Variable


class EnvGenerator(object):
    """
    Class for generating the required gym environment creator
    """

    def __init__(self, name, seed, goal_based=True):
        self.name = name
        self.goal_based = goal_based
        self.seed = seed
        # Create the suitable environment

        self.env = gym.make(name)
        if goal_based:
            self.env = gym.wrappers.FlattenDictWrapper(
                self.env, ['observation', 'desired_goal']
            )

    def make_env_goal_based(self, keys_to_concatenate):
        self.env = gym.wrappers.FlattenDictWrapper(
            self.env, keys_to_concatenate
        )

    def seed_env(self):
        self.env.seed(self.seed)

    def get_environment(self):
        return self.env

    def get_observation_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

    def get_observation_dim(self):
        observation = self.env.reset()['observation']
        return observation.shape[0]

    def get_action_dim(self):
        return self.get_action_space().shape[0]

    def get_goal_dim(self):
        d = self.env.reset()
        d = d['desired_goal']
        return d.shape[0]

    def get_action_shape(self):
        return self.get_action_space().shape

    def get_observation_shape(self):
        return self.get_observation_space().shape

    def take_random_action(self):
        return self.get_action_space().sample()

    def render(self):
        self.env.render()


def plot_goals(rewards, suc):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('reward: %s' % (np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('success')
    plt.plot(suc)
    plt.show()


def plot(x, f_name=None, save_fig=True, show_fig=True):
    fig = plt.figure(figsize=(20, 5))
    plt.plot(x)
    if show_fig:
        plt.show()
    if save_fig:
        if f_name is not None:
            fig.savefig(f_name)


def to_tensor(v, use_cuda=True):
    if use_cuda:
        v = torch.cuda.FloatTensor(v)
    else:
        v = torch.FloatTensor(v)
    return v


def soft_update(polyak_factor, target_network, network):
    """
    Soft update of the parameters using Polyak averaging
    :param polyak_factor: The factor by which to move the averages
    :param target_network: The network to load the weights INTO
    :param network: The network to load weights FROM
    """
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))



def hard_update(target_network, network):
    """
    Hard Update of the networks
    :param target_network: The network to load the weights INTO
    :param network: The network to load weights FROM
    """
    target_network.load_state_dict(network.state_dict())


def get_epsilon_iteration(steps_done, EPS_END, EPS_START, EPS_DECAY):
    """
    Used for the epsilon greedy policy (Used in DQN)
    :param steps_done:
    :param EPS_END:
    :param EPS_START:
    :param EPS_DECAY:
    :return:
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold