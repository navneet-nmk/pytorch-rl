"""
Class for a generic trainer used for training all the different models
"""

import Buffer
import torch
from utils import to_tensor



class Trainer(object):

    def __init__(self, ddpg, batch_size, gamma, num_epochs,
                 num_rollouts, num_eval_rollouts, num_episodes, criterion, learning_rate,
                 polyak_constant, critic_learning_rate, env, nb_train_steps,
                 max_episodes_per_epoch,
                 her_training=False,
                 multi_gpu_training=False,
                 use_cuda=True):

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
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_eval_rollouts = num_eval_rollouts
        self.num_episodes = num_episodes
        self.lr = learning_rate
        self.tau = polyak_constant
        self.critic_lr = critic_learning_rate
        self.env = env
        self.nb_train_steps = nb_train_steps
        self.max_episodes = max_episodes_per_epoch
        self.her = her_training
        self.multi_gpu = multi_gpu_training
        self.cuda = use_cuda

        self.all_rewards = []
        self.successes = []


        # Get the target  and standard networks
        self.target_actor = self.ddpg.get_actors()['target']
        self.actor = self.ddpg.get_actors()['actor']
        self.target_critic  = self.ddpg.get_critics()['target']
        self.critic = self.ddpg.get_critics()['critic']

    def train(self):

        epoch_episode_rewards = []
        epoch_episode_success = []
        epoch_episode_steps = []

        # Initialize the training with an initial state
        state = self.env.reset()

        # Initialize the losses
        loss = 0
        episode_reward =  0
        episode_success = 0
        episode_step = 0
        epoch_actions = []
        t = 0

        # Check whether to use cuda or not
        state = to_tensor(state, use_cuda=self.cuda)
        state = torch.unsqueeze(state, dim=0)

        # Main training loop
        for epoch in range(self.num_epochs):
            for episode in range(self.max_episodes):

                # Rollout of trajectory to fill the replay buffer before training
                for rollout in range(self.num_rollouts):
                    # Predict the nest action
                    action = self.ddpg.get_action(state=state, noise=True)
                    assert action.shape == self.env.get_action_shape

                    # Execute next action
                    new_state, reward, done, success = self.env.step(action)
                    success = success['is_success']
                    done_bool = done * 1

                    t+=1
                    episode_reward += reward
                    episode_step += 1
                    episode_success += success

                    # Book keeping
                    epoch_actions.append(action)
                    # Store the transition in the replay buffer of the agent
                    self.ddpg.store_transition(state=state, new_state=new_state,
                                               action=action, done=done_bool, reward=reward,
                                               success=success)
                    # Set the current state as the next state
                    state = to_tensor(new_state, use_cuda=self.cuda)
                    state = torch.unsqueeze(state, dim=0)

                    # End of the episode
                    if done:
                        epoch_episode_rewards.append(episode_reward)
                        epoch_episode_success.append(episode_success)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0
                        episode_step = 0
                        episode_success = 0

                        # Reset the agent
                        self.ddpg.reset()
                        # Get a new initial state to start from
                        state = self.env.reset()
                        state = to_tensor(state, use_cuda=self.cuda)

                # Train
                epoch_actor_losses = []
                epoch_critic_losses = []

                for train_steps in range(self.nb_train_steps):
                    critic_loss, actor_loss = self.ddpg.fit_batch()
                    if critic_loss is not None and actor_loss is not None:
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)





















        return
