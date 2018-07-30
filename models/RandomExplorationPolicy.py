import torch
import gym
import numpy as np
import os
import scipy.misc as m
from random import randint
import random


class RandomExplorationPolicy(object):

    def __init__(self, env, states_to_save, seed,
                 save_obs=False):

        self.env = env
        self.action = self.env.action_space
        self.obs = self.env.observation_space
        self.seed = seed
        self.save_obs = save_obs
        self.num_actions = self.env.action_space.n
        self.num_states_to_save = states_to_save

    def set_seed(self):
        random.seed = self.seed


    def step(self):

        self.set_seed()
        """

        Executing a random action in the environment

        :return:
        """
        state = self.env.reset()

        for i in range(self.num_states_to_save):
            action = randint(0, self.num_actions-1)
            state, reward, done, success  = env.step(action=action)
            if i % 4 ==0 :
                file_name = str(i) + '.jpg'
                path = os.path.join('montezuma_resources', file_name)
                m.imsave(path, state)

            if done:
                env.reset()



if __name__ == '__main__':
    env  = gym.make('MontezumaRevenge-v0')
    re = RandomExplorationPolicy(env=env, states_to_save=16000, seed=100)
    re.step()
