import gym
import pickle
import os
import scipy.misc as m
from random import randint
import random
import numpy as np
from tqdm import tqdm


class RandomExplorationPolicy(object):

    def __init__(self, env, states_to_save, seed,
                 ram_env=None,
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
        self.ram_env = ram_env
        self.ram_states = []
        self.states = []

    def set_seed(self):
        random.seed = self.seed

    def get_demonstrations(self):
        # Open the demonstration file and store the actions
        if self.demonstrations is not None:
            with open(self.demonstrations, 'rb') as f:
                dat = pickle.load(f)
            self.demonstrated_actions = dat['actions']


    def calc_mean_std(self, images):
        mean_image = np.mean(images, axis=(0, 1))
        std_image = np.std(images, axis=(0, 1))
        return mean_image, std_image

    def step(self, use_demonstrations=False):

        self.set_seed()
        """

        Executing a random action in the environment

        :return:
        """

        state = self.env.reset()
        state_ram = self.ram_env.reset()

        if use_demonstrations:
            self.get_demonstrations()

            for i, a in tqdm(enumerate(self.demonstrated_actions)):
                state, reward, done, success = self.env.step(action=a)
                state_ram, reward_ram, done_ram, success_ram = self.ram_env.step(a)
                if i % 10 == 0:
                    file_name = str(i) + '.jpg'
                    path =  os.path.join('montezuma_resources', file_name)
                    m.imsave(path, state)
                    # Try learning an infogan on the ram states
                    self.ram_states.append(state_ram)
                    self.states.append(state)
            # Save the ram states
            file_name = 'states_ram.npy'
            path = os.path.join('montezuma_resources', file_name)
            np.save(path, self.ram_states)

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
        self.calc_mean_std(self.states)


if __name__ == '__main__':
    env  = gym.make('MontezumaRevenge-v0')
    # Using the RAM Model to get the state representation in the latent vector form
    env_ram = gym.make('MontezumaRevenge-ram-v0')

    re = RandomExplorationPolicy(env=env, states_to_save=16000, seed=100,
                                 ram_env=env_ram,
                                 demo_file='montezuma_resources/MontezumaRevenge.demo')
    re.step(use_demonstrations=True)