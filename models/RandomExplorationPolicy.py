import torch
import gym
import numpy as np


class RandomExplorationPolicy(object):

    def __init__(self, env, action_space,
                 observation_space, seed,
                 save_obs=False):

        self.env = env
        self.action = action_space
        self.obs = observation_space
        self.seed = seed
        self.save_obs = save_obs

    def step(self):
        """

        Executing a random action in the environment

        :return:
        """