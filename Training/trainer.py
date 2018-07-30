"""
Class for a generic trainer used for training all the different reinforcement learning models
"""
import torch
import torch.nn as nn
from Utils.utils import *
from collections import deque, defaultdict
from models.attention import *
import time
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable


class Trainer(object):

    def __init__(self, agent, num_epochs,
                 num_rollouts, num_eval_rollouts, env, eval_env, nb_train_steps,
                 max_episodes_per_epoch, random_seed,
                 output_folder=None,
                 her_training=False,
                 multi_gpu_training=False,
                 use_cuda=True, verbose=True,
                 save_model=False, plot_stats=True, future=None):

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
        self.seed(random_seed)
        self.her = her_training
        self.multi_gpu = multi_gpu_training
        self.cuda = use_cuda
        self.verbose = verbose
        self.plot_stats = plot_stats
        self.save_model = save_model
        self.output_folder = output_folder
        self.future = future

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
            else:
                print("Only 1 gpu available for training .....")

    def train_on_policy(self):
        pass

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

        # Epoch Rewards and success
        epoch_rewards = []
        epoch_success = []

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

            # Log the epoch rewards and successes
            epoch_rewards.append(np.mean(epoch_episode_rewards))
            epoch_success.append(np.mean(epoch_episode_success))

        # Plot the statistics calculated
        if self.plot_stats:
            # Plot the rewards and successes
            rewards_fname = self.output_folder + '/rewards.jpg'
            success_fname = self.output_folder + '/success.jpg'
            plot(epoch_rewards, f_name=rewards_fname, save_fig=True, show_fig=False)
            plot(epoch_success, f_name=success_fname, save_fig=True, show_fig=False)

        # Save the models on the disk
        if self.save_model:
            self.ddpg.save_model(self.output_folder)

        return self.combined_statistics

    def seed(self, s):
        # Seed everything to make things reproducible
        self.env.seed(s)
        np.random.seed(seed=s)
        random.seed = s
        if self.eval_env is not None:
            self.eval_env.seed(s)

    def get_frames(self, transition, sample_experience, k):
        """

        :param transition: Current transition -> Goal substitution
        :param sample_experience: The Future episode experiences
        :param k: The number of transitions to consider
        :return:
        """
        # Get the frames predicted by our self attention network
        seq_length = len(sample_experience)
        states = []
        new_states= []
        rewards = []
        successes = []
        actions = []
        dones = []
        for t in sample_experience:
            state, new_state, reward, success, action, done_bool = t
            state = np.concatenate(state[:self.ddpg.obs_dim], state['achieved_goal'])
            new_state = np.concatenate(new_state[:self.ddpg.obs_dim], new_state['achieved_goal'])
            states.append(state)
            new_states.append(new_state[:self.ddpg.obs_dim])
            rewards.append(reward)
            successes.append(success)
            actions.append(action)
            dones.append(done_bool)

        # Input Sequence consists of n embeddings of states||achieved_goals
        input_sequence = Variable(torch.cat(states))
        # The Query vector is the current state || desired goal
        state, new_state, reward, success, action, done_bool = transition
        query = Variable(state)

        # The Goal Network
        gn = GoalNetwork(input_dim=seq_length, embedding_dim=self.ddpg.input_dim,
                         query_dim=self.ddpg.input_dim, num_hidden=self.ddpg.num_hidden_units,
                         output_features=1, use_additive=True, use_self_attn=True, use_token2token=True,
                         activation=nn.ReLU)

        if self.cuda:
            input_sequence = input_sequence.cuda()
            query = query.cuda()
            gn = gn.cuda()

        scores = gn(input_sequence, query)
        optimizer_gn = optim.Adam(gn.parameters(), lr=self.ddpg.actor_lr)
        optimizer_gn.zero_grad()
        # Dimension of the scores vector is 1 x n
        # Find the top 5 maximum values from the scores vector and their indexes
        values, indices = torch.topk(scores, k, largest=True)
        # Now we have the indices -> Get the corresponding experiences
        top_experiences = []
        for m in indices:
            top_experiences.append(sample_experience[m])

        # Training Step
        TD_error = 0
        for t in top_experiences:
            TD_error += self.ddpg.calc_td_error(t)
        loss = -1 * (TD_error.mean())
        loss.backward()
        # Clamp the gradients to avoid the vanishing gradient problem
        for param in gn.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer_gn.step()

        return top_experiences

    def sample_goals(self,sampling_strategy, experience, future=None, transition=None):
        g = []
        if sampling_strategy == 'final':
            n_s = experience[len(experience)-1]
            g.append(n_s['achieved_goal'])

        elif sampling_strategy == 'self_attention':
            if transition is not None:
                index_of_t = experience.index(transition)
                sample_experience = experience[index_of_t:]
                if future is None:
                    future = 5
                frames = self.get_frames(transition, sample_experience, k=future)
                for f in frames:
                    g.append(f['achieved_goal'])

        elif sampling_strategy == 'future':
            if transition is not None and future is not None:
                index_of_t = transition
                if index_of_t < len(experience)-2:
                    sample_experience = experience[index_of_t+1:]
                    random_transitions = random.sample(population=sample_experience,
                                              k=future)
                    for f in random_transitions:
                        observation, new_observation, state, new_state, reward, success, action, done_bool, achieved_goal, desired_goal = f
                        g.append(achieved_goal)

        elif sampling_strategy == 'prioritized':
            pass

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


        episode_states_history = deque(maxlen=100)
        episode_new_states_history = deque(maxlen=100)

        # Rewards and success for each epoch
        epoch_rewards = []
        epoch_success = []

        # Sample a goal g and an initial state s0
        state = self.env.reset() # The state space includes the observation, achieved_goal and the desired_goal
        observation = state['observation']
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']
        state = np.concatenate((observation, desired_goal))

        # If eval, initialize the evaluation with an initial state
        if self.eval_env is not None:
            eval_state = self.eval_env.reset()
            eval_observation = eval_state['observation']
            eval_achieved_goal = eval_state['achieved_goal']
            eval_desired_goal = eval_state['desired_goal']
            eval_state = np.concatenate((eval_observation, eval_desired_goal))
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

            for cycle in range(self.max_episodes):

                # States and new states for the hindsight experience replay
                episode_states = []
                episode_achieved_goals = []
                episode_desired_goals = []
                episode_new_states = []
                episode_rewards = []
                episode_successes = []
                episode_actions = []
                episode_dones = []
                episode_experience = []
                episode_observations = []
                episode_new_observations = []

                # Rollout of trajectory to fill the replay buffer before the training
                for rollout in range(self.num_rollouts):
                    # Sample an action from behavioural policy pi
                    action = self.ddpg.get_action(state=state, noise=True)
                    #assert action.shape == self.env.get_action_shape

                    # Execute the action and observe the new state
                    new_state, reward, done, success = self.env.step(action)

                    # The following has to hold
                    assert reward == self.env.compute_reward(
                        new_state['achieved_goal'], new_state['desired_goal'],
                        info=success
                    )

                    new_observation = new_state['observation']
                    new_achieved_goal = new_state['achieved_goal']
                    new_desired_goal = new_state['desired_goal']
                    new_state = np.concatenate((new_observation, new_desired_goal))
                    new_state = to_tensor(new_state, self.cuda)
                    #new_state = torch.unsqueeze(new_state, dim=0)
                    success = success['is_success']
                    done_bool = done * 1

                    episode_states.append(state)
                    episode_new_states.append(new_state)
                    episode_rewards.append(reward)
                    episode_successes.append(success)
                    episode_actions.append(action)
                    episode_dones.append(done_bool)
                    episode_achieved_goals.append(new_achieved_goal)
                    episode_desired_goals.append(new_desired_goal)
                    episode_observations.append(observation)
                    episode_new_observations.append(new_observation)
                    episode_experience.append(
                        (observation, new_observation, state, new_state, reward, success, action, done_bool, new_achieved_goal, desired_goal)
                    )

                    t += 1
                    episode_reward += reward
                    episode_step += 1
                    episode_success += success

                    # Set the current state as the next state
                    state = to_tensor(new_state, use_cuda=self.cuda)
                    state = torch.unsqueeze(state, dim=0)
                    observation = new_observation

                    # End of the episode
                    if done:
                        # Get the episode goal
                        #episode_goal = new_state[:self.ddpg.obs_dim]
                        #episode_goals_history.append(episode_goal)
                        epoch_episode_rewards.append(episode_reward)
                        episode_goals_history.append(achieved_goal)
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
                        observation = state['observation']
                        achieved_goal = state['achieved_goal']
                        desired_goal = state['desired_goal']
                        state = np.concatenate((observation, desired_goal))
                        state = to_tensor(state, use_cuda=self.cuda)
                        state = torch.unsqueeze(state, dim=0)

                # Standard Experience Replay
                i = 0
                for tr in episode_experience:
                    observation, new_observation, state, new_state, reward, success, action, done_bool, achieved_goal, desired_goal = tr
                    new_state = torch.unsqueeze(new_state, dim=0)
                    action = to_tensor(action, use_cuda=self.cuda)
                    action = torch.unsqueeze(action, dim=0)
                    reward = to_tensor([np.asscalar(reward)], use_cuda=self.cuda)
                    done_bool = to_tensor([done_bool], use_cuda=self.cuda)
                    #success = to_tensor([np.asscalar(success)], use_cuda=self.cuda)

                    # Store the transition in the experience replay
                    self.ddpg.store_transition(
                        state=state, new_state=new_state, reward=reward,
                        success=success, action=action, done=done_bool
                    )

                    # Hindsight Experience Replay
                    # Sample a set of additional goals for replay G: S
                    additional_goals = self.sample_goals(sampling_strategy='future',
                                                         experience=episode_experience,
                                                         future=self.future, transition=i)

                    for g in additional_goals:
                        # Recalculate the reward
                        substitute_goal = g

                        # Recalculate the reward now when the desired goal is the substituted goal
                        # which is the achieved goal sampled using the sampling strategy
                        reward_revised = self.env.compute_reward(
                            achieved_goal, substitute_goal, info=success
                        )
                        # Book Keeping
                        #episode_revised_rewards_history.append(reward_revised)
                        # Store the transition with the new goal and reward in the replay buffer
                        # Get the observation and new observation from the concatenated value

                        # Currently, the env on resetting returns a concatenated vector of
                        # Observation and the desired goal. Therefore, we need to extract the
                        # Observation for this step.
                        observation = to_tensor(observation, use_cuda=self.cuda)
                        new_observation = to_tensor(new_observation, use_cuda=self.cuda)

                        g = to_tensor(g, use_cuda=self.cuda)
                        #reward_revised = to_tensor(reward_revised, use_cuda=self.cuda)
                        #print(observation)
                        augmented_state = torch.cat([observation, g])
                        augmented_new_state = torch.cat([new_observation, g])
                        augmented_state = torch.unsqueeze(augmented_state, dim=0)
                        augmented_new_state = torch.unsqueeze(augmented_new_state, dim=0)
                        reward_revised = to_tensor([np.asscalar(reward_revised)], use_cuda=self.cuda)

                        # Store the transition in the buffer
                        self.ddpg.store_transition(state=augmented_state, new_state=augmented_new_state,
                                                   action=action, done=done_bool, reward=reward_revised,
                                                   success=success)

                # Train the network
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
                        eval_episode_success += eval_success['is_success']

                        if eval_done:
                            # Get the episode goal
                            #eval_episode_goal = eval_new_state[:self.ddpg.obs_dim]
                            #eval_episode_goals_history.append(eval_episode_goal)
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
                    print(epoch)
                    print("Reward ", statistics['rollout/rewards'])
                    print("Successes ", statistics['rollout/successes'])

                    if self.eval_env is not None:
                        print("Evaluation Reward ", statistics['eval/rewards'])
                        print("Evaluation Successes ", statistics['eval/successes'])

            # Log the combined statistics for all epochs
            for key in sorted(statistics.keys()):
                self.combined_statistics[key].append(statistics[key])

            # Log the epoch rewards and successes
            epoch_rewards.append(np.mean(epoch_episode_rewards))
            epoch_success.append(np.mean(epoch_episode_success))

        # Plot the statistics calculated
        if self.plot_stats:
            # Plot the rewards and successes
            rewards_fname = self.output_folder + '/rewards.jpg'
            success_fname = self.output_folder + '/success.jpg'
            plot(epoch_rewards, f_name=rewards_fname, save_fig=True, show_fig=False)
            plot(epoch_success, f_name=success_fname, save_fig=True, show_fig=False)

        # Save the models on the disk
        if self.save_model:
            self.ddpg.save_model(self.output_folder)

        return self.combined_statistics





