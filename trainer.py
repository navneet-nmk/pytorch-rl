"""
Class for a generic trainer used for training all the different models
"""

import Buffer


class Trainer(object):

    def __init__(self, ddpg,batch_size, buffer_size, gamma, num_epochs,
                 num_rollouts,num_episodes, criterion, learning_rate,
                 polyak_constant, critic_learning_rate, her_training=False):

        """

        :param ddpg: the ddpg network
        :param batch_size: size of the batch of samples
        :param buffer_size: size of the experience replay
        :param gamma: discount factor to account for future rewards
        :param num_rollouts: number of experience gatthering rollouts per episode
        :param num_episodes: number of episodes per epoch
        :param criterion: loss function
        :param polyak_constant: the moving average value (used in polyak averaging)
        :param her_training: use hindsight experience replay
        """
        self.ddpg = ddpg
        self.criterion = criterion
        self.batch_size = batch_size
        self.buffer = Buffer.ReplayBuffer(capacity=buffer_size)
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_episodes = num_episodes
        self.lr = learning_rate
        self.tau = polyak_constant
        self.critic_lr = critic_learning_rate
        self.her = her_training
        self.all_rewards = []
        self.successes = []

        # Get the target  and standard networks
        self.target_actor = self.ddpg.get_actors()['target']
        self.actor = self.ddpg.get_actors()['actor']
        self.target_critic  = self.ddpg.get_critics()['target']
        self.critic = self.ddpg.get_critics()['critic']

    def train(self):

        return
