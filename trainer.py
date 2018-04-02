"""
Class for a generic trainer used for training all the different models
"""

import Buffer


class Trainer(object):

    def __init__(self, target_actor, target_critic, actor, critic,
                 batch_size, buffer_size, gamma, num_epochs, num_rollouts,
                 num_episodes, criterion, learning_rate, critic_learning_rate,
                 her_training=False):

        """

        :param target_actor: Target Actor network (Stabilizes training)
        :param target_critic: Target Critic network (Stabilizes training)
        :param actor: Actor network used in training
        :param critic: critic used in training
        :param batch_size: size of the batch of samples
        :param buffer_size: size of the experience replay
        :param gamma: discount factor to account for future rewards
        :param num_rollouts: number of experience gatthering rollouts per episode
        :param num_episodes: number of episodes per epoch
        :param criterion: loss function
        :param her_training: use hindsight experience replay
        """

        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor = actor
        self.critic = critic
        self.criterion = criterion
        self.batch_size = batch_size
        self.buffer = Buffer.ReplayBuffer(capacity=buffer_size)
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_episodes = num_episodes
        self.lr = learning_rate
        self.critic_lr = critic_learning_rate
        self.her = her_training

    def train(self):
        return
