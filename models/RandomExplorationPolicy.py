import gym
import pickle
import os
import scipy.misc as m
from random import randint
import random


class RandomExplorationPolicy(object):

    def __init__(self, env, states_to_save, seed,
                 demo_file=None,
                 save_obs=False):

        self.env = env
        self.action = self.env.action_space
        self.obs = self.env.observation_space
        self.seed = seed
        self.save_obs = save_obs
        self.num_actions = self.env.action_space.n
        self.num_states_to_save = states_to_save
        self.demonstrations = demo_file
        self.demonstrated_actions = []

    def set_seed(self):
        random.seed = self.seed

    def get_demonstrations(self):
        # Open the demonstration file and store the actions
        if self.demonstrations is not None:
            with open(self.demonstrations, 'rb') as f:
                dat = pickle.load(f)
            self.demonstrated_actions = dat['actions']

    def step(self, use_demonstrations=False):

        self.set_seed()
        """

        Executing a random action in the environment

        :return:
        """

        state = self.env.reset()

        if use_demonstrations:
            self.get_demonstrations()
            for i, a in enumerate(self.demonstrated_actions):
                state, reward, done, success = self.env.step(a)
                if i%10 == 0:
                    file_name = str(i) + '.jpg'
                    path =  os.path.join('montezuma_resources', file_name)
                    m.imsave(path, state)
                if done:
                    state = self.env.reset()

        else:
            for i in range(self.num_states_to_save):
                action = randint(0, self.num_actions-1)
                state, reward, done, success  = self.env.step(action=action)
                if i % 4 ==0 :
                    file_name = str(i) + '.jpg'
                    path = os.path.join('montezuma_resources', file_name)
                    m.imsave(path, state)

                if done:
                    self.env.reset()



if __name__ == '__main__':
    env  = gym.make('MontezumaRevenge-v0')
    re = RandomExplorationPolicy(env=env, states_to_save=16000, seed=100, demo_file='montezuma_resources/MontezumaRevenge.demo')
    re.step(use_demonstrations=True)
