"""
Class for a generic trainer used for training all the different models
"""
import torch
import torch.nn as nn
from utils import to_tensor, plot_rewards, plot_success
from collections import deque, defaultdict
import time
import numpy as np


class Trainer(object):

    def __init__(self, agent, num_epochs,
                 num_rollouts, num_eval_rollouts, env, eval_env, nb_train_steps,
                 max_episodes_per_epoch,
                 output_folder=None,
                 her_training=False,
                 multi_gpu_training=False,
                 use_cuda=True, verbose=True,
                 save_model=False, plot_stats=True):

        """

        :param ddpg: The ddpg network
        :param num_rollouts: number of experience gathering rollouts per episode
        :param num_eval_rollouts: number of evaluation rollouts
        :param num_episodes: number of episodes per epoch
        :param env: Gym environment to train on
        :param eval_env: Gym environment to evaluate on
        :param nb_train_steps: training steps to take
        :param max_episodes_per_epoch: maximum number of episodes per epoch
        :param her_training: use hindsight experience replay
        :param multi_gpu_training: train on multiple gpus
        """

        self.ddpg = agent
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_eval_rollouts = num_eval_rollouts
        self.env = env
        self.eval_env = eval_env
        self.nb_train_steps = nb_train_steps
        self.max_episodes = max_episodes_per_epoch
        self.her = her_training
        self.multi_gpu = multi_gpu_training
        self.cuda = use_cuda
        self.verbose = verbose
        self.plot_stats = plot_stats
        self.save_model = save_model
        self.output_folder = output_folder

        self.all_rewards = []
        self.successes = []

        # Get the target  and standard networks
        self.target_actor = self.ddpg.get_actors()['target']
        self.actor = self.ddpg.get_actors()['actor']
        self.target_critic  = self.ddpg.get_critics()['target']
        self.critic = self.ddpg.get_critics()['critic']
        self.statistics = defaultdict(float)
        self.combined_statistics = defaultdict(list)

        if self.multi_gpu:
            if torch.cuda.device_count() > 1:
                print("Training on ", torch.cuda.device_count() , " GPUs ")
                self.target_critic = nn.DataParallel(self.target_critic)
                self.critic = nn.DataParallel(self.critic)
                self.target_actor = nn.DataParallel(self.target_actor)
                self.actor = nn.DataParallel(self.actor)

    def train(self):

        # Starting time
        start_time = time.time()

        # Initialize the statistics dictionary
        statistics = self.statistics

        episode_rewards_history = deque(maxlen=100)
        eval_episode_rewards_history = deque(maxlen=100)
        episode_success_history = deque(maxlen=100)
        eval_episode_success_history = deque(maxlen=100)

        epoch_episode_rewards = []
        epoch_episode_success = []
        epoch_episode_steps = []

        # Initialize the training with an initial state
        state = self.env.reset()
        # If eval, initialize the evaluation with an initial state
        if self.eval_env is not None:
            eval_state = self.eval_env.reset()
            eval_state = to_tensor(eval_state, use_cuda=self.cuda)
            eval_state = torch.unsqueeze(eval_state, dim=0)

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
            epoch_actor_losses = []
            epoch_critic_losses = []
            for episode in range(self.max_episodes):

                # Rollout of trajectory to fill the replay buffer before training
                for rollout in range(self.num_rollouts):
                    # Sample an action from behavioural policy pi
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
                        episode_rewards_history.append(episode_reward)
                        episode_success_history.append(episode_success)
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
                for train_steps in range(self.nb_train_steps):
                    critic_loss, actor_loss = self.ddpg.fit_batch()
                    if critic_loss is not None and actor_loss is not None:
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)

                    # Update the target networks using polyak averaging
                    self.ddpg.update_target_networks()

                eval_episode_rewards = []
                eval_episode_successes = []
                if self.eval_env is not None:
                    eval_episode_reward = 0
                    eval_episode_success = 0
                    for t_rollout in range(self.num_eval_rollouts):
                        if eval_state is not None:
                            eval_action = self.ddpg.get_action(state=eval_state, noise=False)
                        eval_new_state, eval_reward, eval_done, eval_success = self.eval_env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_episode_success += eval_success

                        if eval_done:
                            eval_state = self.eval_env.reset()
                            eval_state = to_tensor(eval_state, use_cuda=self.cuda)
                            eval_state = torch.unsqueeze(eval_state, dim=0)
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_successes.append(eval_episode_success)
                            eval_episode_success_history.append(eval_episode_success)
                            eval_episode_reward = 0
                            eval_episode_success = 0

            # Log stats
            duration = time.time() - start_time
            statistics['rollout/rewards'] = np.mean(epoch_episode_rewards)
            statistics['rollout/rewards_history'] = np.mean(episode_rewards_history)
            statistics['rollout/successes'] = np.mean(epoch_episode_success)
            statistics['rollout/successes_history'] = np.mean(episode_success_history)
            statistics['rollout/actions_mean'] = np.mean(epoch_actions)
            statistics['train/loss_actor'] = np.mean(epoch_actor_losses)
            statistics['train/loss_critic'] = np.mean(epoch_critic_losses)
            statistics['total/duration'] = duration

            # Evaluation statistics
            if self.eval_env is not None:
                statistics['eval/rewards'] = np.mean(eval_episode_rewards)
                statistics['eval/rewards_history'] = np.mean(eval_episode_rewards_history)
                statistics['eval/successes'] = np.mean(eval_episode_successes)
                statistics['eval/success_history'] = np.mean(eval_episode_success_history)

            # Print the statistics
            if self.verbose:
                if epoch % 5 == 0:
                    print("Actor Loss: ", statistics['train/loss_actor'])
                    print("Critic Loss: ", statistics['train/loss_critic'])
                    print("Reward ", statistics['rollout/rewards'])
                    print("Successes ", statistics['rollout/successes'])

                    if self.eval_env is not None:
                        print("Evaluation Reward ", statistics['eval/rewards'])
                        print("Evaluation Successes ", statistics['eval/successes'])

            # Log the combined statistics for all epochs
            for key in sorted(statistics.keys()):
                self.combined_statistics[key].append(statistics[key])

        # Plot the statistics calculated
        if self.plot_stats:
            # Plot the rewards and successes
            plot_rewards(self.combined_statistics['rollout/rewards_history'])
            plot_success(self.combined_statistics['rollout/successes_history'])

        # Save the models on the disk
        if self.save_model:
            self.ddpg.save_model(self.output_folder)

        return self.combined_statistics

    def sample_goals(self, sampling_strategy=None):
        g = 0
        return g

    def her_training(self):

        # Starting Time
        start_time = time.time()

        # Initialize the statistics dictionary
        statistics = self.statistics

        episode_rewards_history = deque(maxlen=100)
        episode_revised_rewards_history  =  deque(maxlen=100)
        eval_episode_rewards_history = deque(maxlen=100)
        episode_success_history = deque(maxlen=100)
        eval_episode_success_history = deque(maxlen=100)
        episode_goals_history = deque(maxlen=100)
        eval_episode_goals_history = deque(maxlen=100)
        all_goals_history = deque(maxlen=100)

        epoch_episode_rewards = []
        epoch_episode_success = []
        epoch_episode_steps = []

        # Sample a goal g and an initial state s0
        state = self.env.reset() # The state space includes the observation as well as the desired goal

        # If eval, initialize the evaluation with an initial state
        if self.eval_env is not None:
            eval_state = self.eval_env.reset()
            eval_state = to_tensor(eval_state, use_cuda=self.cuda)
            eval_state = torch.unsqueeze(eval_state, dim=0)

        # Initialize the losses
        loss = 0
        episode_reward = 0
        episode_success = 0
        episode_step = 0
        epoch_actions = []
        t = 0

        # Check whether to use cuda or not
        state = to_tensor(state, use_cuda=self.cuda)
        state = torch.unsqueeze(state, dim=0)

        for epoch in range(self.num_epochs):
            epoch_actor_losses = []
            epoch_critic_losses = []

            for episode in range(self.max_episodes):

                # Rollout of trajectory to fill the replay buffer before the training
                for rollout in range(self.num_rollouts):
                    # Sample an action from behavioural policy pi
                    action = self.ddpg.get_action(state=state, noise=True)
                    assert action.shape == self.env.get_action_shape

                    # Execute the action and observe the new state
                    new_state, reward, done, success = self.env.step(action)

                    # The following has to hold
                    assert reward == self.env.compute_reward(
                        new_state['achieved_goal'], new_state['desired_goal'],
                        info=success
                    )

                    new_state = to_tensor(new_state, self.cuda)
                    success = success['is_success']
                    done_bool = done * 1

                    t += 1
                    episode_reward += reward
                    episode_step += 1
                    episode_success += success

                    # Book keeping
                    epoch_actions.append(action)
                    # Store the transition in the replay buffer of the agent
                    self.ddpg.store_transition(state=state, new_state=new_state,
                                               action=action, done=done_bool, reward=reward,
                                               success=success)

                    # Sample a set of additional goals for replay G: S(current episode)
                    additional_goals = self.sample_goals()
                    for goal in additional_goals:
                        # Recalculate the reward

                        # Substitute goals
                        substitute_goal = new_state['achieved_goal'].copy()
                        reward_revised = self.env.compute_reward(
                            new_state['achieved_goal'], substitute_goal, info=success
                        )
                        # Book Keeping
                        episode_revised_rewards_history.append(reward_revised)
                        # Store the transition with the new goal and reward in the replay buffer
                        # Get the observation and new observation from the concatenated value

                        # Currently, the env on resetting returns a concatenated vector of
                        # Observation and the desired goal. Therefore, we need to extract the
                        # Observation for this step.

                        state = state[:self.ddpg.obs_dim]
                        new_state = new_state[:self.ddpg.obs_dim]

                        old_goal = state[self.ddpg.obs_dim:] # For book keeping
                        new_goal = new_state[self.ddpg.obs_dim:]

                        all_goals_history.append(old_goal)
                        if new_goal != old_goal:
                            all_goals_history.append(new_goal)

                        augmented_state = torch.cat([state, goal])
                        augmented_new_state = torch.cat([new_state, goal])

                        # Store the transition in the buffer
                        self.ddpg.store_transition(state=augmented_state, new_state=augmented_new_state,
                                                   action=action, done=done_bool, reward=reward_revised,
                                                   success=success)


                    # Set the current state as the next state
                    state = to_tensor(new_state, use_cuda=self.cuda)
                    state = torch.unsqueeze(state, dim=0)

                    # End of the episode
                    if done:
                        # Get the episode goal
                        episode_goal = new_state[:self.ddpg.obs_dim]
                        episode_goals_history.append(episode_goal)
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        episode_success_history.append(episode_success)
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



                    # Train the network
                    # Train
                    for train_steps in range(self.nb_train_steps):
                        critic_loss, actor_loss = self.ddpg.fit_batch()
                        if critic_loss is not None and actor_loss is not None:
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)

                        # Update the target networks using polyak averaging
                        self.ddpg.update_target_networks()

                    eval_episode_rewards = []
                    eval_episode_successes = []
                    if self.eval_env is not None:
                        eval_episode_reward = 0
                        eval_episode_success = 0
                        for t_rollout in range(self.num_eval_rollouts):
                            if eval_state is not None:
                                eval_action = self.ddpg.get_action(state=eval_state, noise=False)
                            eval_new_state, eval_reward, eval_done, eval_success = self.eval_env.step(eval_action)
                            eval_episode_reward += eval_reward
                            eval_episode_success += eval_success

                            if eval_done:
                                # Get the episode goal
                                eval_episode_goal = eval_new_state[:self.ddpg.obs_dim]
                                eval_episode_goals_history.append(eval_episode_goal)
                                eval_state = self.eval_env.reset()
                                eval_state = to_tensor(eval_state, use_cuda=self.cuda)
                                eval_state = torch.unsqueeze(eval_state, dim=0)
                                eval_episode_rewards.append(eval_episode_reward)
                                eval_episode_rewards_history.append(eval_episode_reward)
                                eval_episode_successes.append(eval_episode_success)
                                eval_episode_success_history.append(eval_episode_success)
                                eval_episode_reward = 0
                                eval_episode_success = 0

                # Log stats
                duration = time.time() - start_time
                statistics['rollout/rewards'] = np.mean(epoch_episode_rewards)
                statistics['rollout/rewards_history'] = np.mean(episode_rewards_history)
                statistics['rollout/successes'] = np.mean(epoch_episode_success)
                statistics['rollout/successes_history'] = np.mean(episode_success_history)
                statistics['rollout/actions_mean'] = np.mean(epoch_actions)
                statistics['rollout/goals_mean'] = np.mean(episode_goals_history)
                statistics['train/loss_actor'] = np.mean(epoch_actor_losses)
                statistics['train/loss_critic'] = np.mean(epoch_critic_losses)
                statistics['total/duration'] = duration

                # Evaluation statistics
                if self.eval_env is not None:
                    statistics['eval/rewards'] = np.mean(eval_episode_rewards)
                    statistics['eval/rewards_history'] = np.mean(eval_episode_rewards_history)
                    statistics['eval/successes'] = np.mean(eval_episode_successes)
                    statistics['eval/success_history'] = np.mean(eval_episode_success_history)
                    statistics['eval/goals_history'] = np.mean(eval_episode_goals_history)

                # Print the statistics
                if self.verbose:
                    if epoch % 5 == 0:
                        print("Actor Loss: ", statistics['train/loss_actor'])
                        print("Critic Loss: ", statistics['train/loss_critic'])
                        print("Reward ", statistics['rollout/rewards'])
                        print("Successes ", statistics['rollout/successes'])

                        if self.eval_env is not None:
                            print("Evaluation Reward ", statistics['eval/rewards'])
                            print("Evaluation Successes ", statistics['eval/successes'])

                # Log the combined statistics for all epochs
                for key in sorted(statistics.keys()):
                    self.combined_statistics[key].append(statistics[key])

            # Plot the statistics calculated
            if self.plot_stats:
                # Plot the rewards and successes
                plot_rewards(self.combined_statistics['rollout/rewards_history'])
                plot_success(self.combined_statistics['rollout/successes_history'])

            # Save the models on the disk
            if self.save_model:
                self.ddpg.save_model(self.output_folder)

            return self.combined_statistics

























