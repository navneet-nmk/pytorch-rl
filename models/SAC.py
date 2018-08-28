"""

This script contains an implementation of the Soft Actor Critic.

This is an off policy Actor critic algorithm with the entropy of
the current policy added to the reward.

The maximization of the augmented reward enables the agent to discover multimodal actions
(Multiple actions that results in reward). It promotes exploration
and is considerably more stable to random seeds variation and hyperparameter
tuning compared to Deep Deterministic Policy Gradient.

"""

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

import torch.nn as nn
import torch.optim as optim
from Memory import Buffer
import Utils.random_process as random_process
from Distributions.distributions import SigmoidNormal
from collections import defaultdict, deque
import time
from Utils.utils import *
from Layers.LayerNorm import LayerNorm

USE_CUDA = torch.cuda.is_available()


class SAC(object):

    def __init__(self, state_dim,
                 action_dim,
                 hidden_dim,
                 actor, critic, value_network,
                 target_value_network,
                 polyak_constant,
                 actor_learning_rate,
                 critic_learning_rate,
                 value_learning_rate,
                 num_q_value,
                 num_v_value,
                 batch_size, gamma,
                 random_seed,
                 num_epochs,
                 num_rollouts, num_eval_rollouts,
                 env, eval_env, nb_train_steps,
                 max_episodes_per_epoch,
                 output_folder,
                 use_cuda, buffer_capacity,
                 policy_reg_mean_weight=1e-3,
                 policy_reg_std_weight=1e-3,
                 policy_preactivation_weight=0,
                 verbose=True,
                 plot_stats=False,
                 ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.q_dim = num_q_value
        self.v_dim = num_v_value
        self.actor = actor
        self.critic = critic
        self.value = value_network
        self.tau = polyak_constant
        self.bs = batch_size
        self.gamma = gamma
        self.seed = random_seed
        self.use_cuda = use_cuda
        self.buffer = Buffer.ReplayBuffer(capacity=buffer_capacity, seed=self.seed)
        self.policy_reg_mean_weight = policy_reg_mean_weight
        self.policy_reg_std_weight = policy_reg_std_weight
        self.policy_pre_activation_weight = policy_preactivation_weight

        # Training specific parameters
        self.num_epochs = num_epochs
        self.num_rollouts = num_rollouts
        self.num_eval_rollouts = num_eval_rollouts
        self.env = env
        self.eval_env = eval_env
        self.nb_train_steps = nb_train_steps
        self.max_episodes_per_epoch = max_episodes_per_epoch
        self.statistics = defaultdict(float)
        self.combined_statistics = defaultdict(list)
        self.verbose = verbose
        self.output_folder = output_folder
        self.plot_stats = plot_stats


        self.actor_optim = optim.Adam(lr=actor_learning_rate, params=self.actor.parameters())
        self.critic_optim = optim.Adam(lr=critic_learning_rate, params=self.critic.parameters())
        self.value_optim = optim.Adam(lr=value_learning_rate, params=self.value.parameters())

        self.target_value = target_value_network

        if self.use_cuda:
            self.actor  = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.value = self.value.cuda()
            self.target_value = self.target_value.cuda()

        # Initializing the target networks with the standard network weights
        self.target_value.load_state_dict(self.value.state_dict())

        # Initialize a random exploration noise
        self.random_noise = random_process.OrnsteinUhlenbeckActionNoise(self.action_dim)

    def save_model(self, output):
        """
        Saving the models
        :param output:
        :return:
        """
        print("Saving the actor, critic and value networks")
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pt'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pt'.format(output)
        )

        torch.save(
            self.value.state_dict(),
            '{}/value.pt'.format(output)
        )

    # Get the action with an option for random exploration noise
    def get_action(self, state, noise=True):
        state_v = Variable(state)
        action = self.actor(state_v)
        if noise:
            noise = self.random_noise
            action = action.data.cpu().numpy()[0] + noise.sample()
        else:
            action = action.data.cpu().numpy()[0]
        action = np.clip(action, -1., 1.)
        return action

    # Reset the noise
    def reset(self):
        self.random_noise.reset()

    # Store the transition into the replay buffer
    def store_transition(self, state, new_state, action, reward, done, success):
        self.buffer.push(state, action, new_state, reward, done, success)

    # Update the target networks using polyak averaging
    def update_target_networks(self):
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def random_action(self):
        """
        Take a random action bounded between min and max values of the action space
        :return:
        """
        action = np.random.uniform(-1., 1., self.action_dim)
        self.a_t = action

        return action

    def seed(self, s):
        """
        Setting the random seed for a particular training iteration
        :param s:
        :return:
        """
        np.random.seed(s)
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)

    # Calculate the Value function error
    def calc_soft_value_function_error(self, states):
        # The values for the states
        values = self.value(states)
        actions, means, log_probs, stds, log_stds, pre_sigmoid_value = self.actor(states)
        q_values = self.critic(states, actions)
        real_values = q_values - log_probs
        real_values.detach()
        loss = nn.MSELoss()(values, real_values)
        return loss, values

    # Calculate the Q function error
    def calc_soft_q_function_error(self, states, actions, next_states, rewards, dones):
        r = rewards
        value_next_states = self.target_value(next_states)
        value_next_states = value_next_states * (1-dones)

        y = r + self.gamma*value_next_states
        y.detach()

        outputs = self.critic(states, actions)
        temporal_difference_loss = nn.MSELoss()(outputs, y)

        return temporal_difference_loss, outputs

    # Calculate the policy loss
    def calc_policy_loss(self, states, q_values, value_predictions):
        actions, means, log_probs, stds, log_stds, pre_sigmoid_value = self.actor(states)
        log_policy_target = q_values - value_predictions
        policy_loss = (
            log_probs * (log_probs - log_policy_target).detach()
        ).mean()
        mean_reg_loss = self.policy_reg_mean_weight * (means**2).mean()
        std_reg_loss = self.policy_reg_std_weight * (log_stds ** 2).mean()

        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_sigmoid_value ** 2).sum(dim=1).mean()
        )

        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

        policy_loss = policy_loss + policy_reg_loss

        return policy_loss

    # Fitting one batch of data from the replay buffer
    def fit_batch(self):
        transitions = self.buffer.sample_batch(self.bs)
        batch = Buffer.Transition(*zip(*transitions))
        # Get the separate values from the named tuple
        states = batch.state
        new_states = batch.next_state
        actions = batch.action
        rewards = batch.reward
        dones = batch.done

        states = Variable(torch.cat(states))
        with torch.no_grad():
            new_states = Variable(torch.cat(new_states))
        actions = Variable(torch.cat(actions))
        rewards = Variable(torch.cat(rewards))
        dones = Variable(torch.cat(dones))

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            new_states = new_states.cuda()
            dones = dones.cuda()

        value_loss, values = self.calc_soft_value_function_error(states)
        q_loss, q_values = self.calc_soft_q_function_error(states, actions, new_states, rewards, dones)
        policy_loss = self.calc_policy_loss(states, q_values, values)

        """
        Update the networks
        """
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.zero_grad()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Update the target networks
        self.update_target_networks()

        return value_loss, q_loss, policy_loss

    # The main training loop
    def train(self):

        for name, param in self.actor.named_parameters():
            print(name, param)

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
        state = to_tensor(state, use_cuda=self.use_cuda)
        state = torch.unsqueeze(state, dim=0)

        # If eval, initialize the evaluation with an initial state
        if self.eval_env is not None:
            eval_state = self.eval_env.reset()
            eval_state = to_tensor(eval_state, use_cuda=self.use_cuda)
            eval_state = torch.unsqueeze(eval_state, dim=0)

        # Initialize the losses
        loss = 0
        episode_reward =  0
        episode_success = 0
        episode_step = 0
        epoch_actions = []
        t = 0

        # Main training loop
        for epoch in range(self.num_epochs):
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_value_losses = []
            for episode in range(self.max_episodes_per_epoch):
                # Rollout of trajectory to fill the replay buffer before training
                for rollout in range(self.num_rollouts):
                    # Sample an action from behavioural policy pi
                    action, means, log_probs, stds, log_stds, pre_sigmoid_value = self.actor(state)
                    action = action.data.cpu().numpy()[0]
                    # Execute next action
                    new_state, reward, done, success = self.env.step(action)
                    done_bool = done * 1

                    # Convert new state to a tensor
                    new_state = to_tensor(new_state, use_cuda=self.use_cuda)
                    new_state = torch.unsqueeze(new_state, dim=0)
                    action = to_tensor(action, use_cuda=self.use_cuda)
                    action = torch.unsqueeze(action, dim=0)

                    t += 1
                    episode_reward += reward
                    episode_step += 1

                    reward = [reward]
                    done_bool = [done_bool]

                    reward = to_tensor(reward, use_cuda=self.use_cuda)
                    done_bool = to_tensor(done_bool, use_cuda=self.use_cuda)


                    # Book keeping
                    epoch_actions.append(action)

                    # Store the transition in the replay buffer of the agent
                    self.store_transition(state=state, new_state=new_state,
                                               action=action, done=done_bool, reward=reward, success=success)

                    # Set the current state as the next state
                    state = to_tensor(new_state, use_cuda=self.use_cuda)

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
                        self.reset()
                        # Get a new initial state to start from
                        state = self.env.reset()
                        state = to_tensor(state, use_cuda=self.use_cuda)
                        state = torch.unsqueeze(state, dim=0)

                # Train
                for train_steps in range(self.nb_train_steps):
                    value_loss, critic_loss, actor_loss = self.fit_batch()
                    if critic_loss is not None and actor_loss is not None and value_loss is not None:
                        epoch_critic_losses.append(critic_loss.data.numpy())
                        epoch_actor_losses.append(actor_loss.data.numpy())
                        epoch_value_losses.append(value_loss.data.numpy())

                # Evaluation Step
                eval_episode_rewards = []
                eval_episode_successes = []
                if self.eval_env is not None:
                    eval_episode_reward = 0
                    eval_episode_success = 0
                    for t_rollout in range(self.num_eval_rollouts):
                        if eval_state is not None:
                            action, means, log_probs, stds, log_stds, pre_sigmoid_value = self.actor(eval_state)
                            eval_new_state, eval_reward, eval_done, eval_success = self.eval_env.step(action)
                            eval_episode_reward += eval_reward
                            eval_episode_success += eval_success

                            if eval_done:
                                eval_state = self.eval_env.reset()
                                eval_state = to_tensor(eval_state, use_cuda=self.use_cuda)
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
            statistics['train/loss_value'] = np.mean(epoch_value_losses)
            statistics['total/duration'] = duration

            # Evaluation statistics
            if self.eval_env is not None:
                statistics['eval/rewards'] = np.mean(eval_episode_rewards)
                statistics['eval/rewards_history'] = np.mean(eval_episode_rewards_history)
                statistics['eval/successes'] = np.mean(eval_episode_successes)
                statistics['eval/success_history'] = np.mean(eval_episode_success_history)

            # Print the statistics
            if self.verbose:
                if epoch % 1 == 0:
                    print("Actor Loss: ", statistics['train/loss_actor'])
                    print("Critic Loss: ", statistics['train/loss_critic'])
                    print("Value Loss: ", statistics['train/loss_value'])
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
            self.save_model(self.output_folder)

        return self.combined_statistics


# The Policy Network
class StochasticActor(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_dim, use_tanh=False,
                 use_sigmoid=False, deterministic=False, use_layernorm=False):
        super(StochasticActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.use_tanh = use_tanh
        self.use_sigmoid = use_sigmoid
        self.deterministic = deterministic
        self.use_layernorm = use_layernorm

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
        self.hidden_2 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden*2)
        self.output_mu = nn.Linear(in_features=self.hidden*2, out_features=self.action_dim)
        self.output_logstd = nn.Linear(in_features=self.hidden*2, out_features=self.action_dim)

        if self.use_layernorm:
            self.ln1 =  LayerNorm(self.hidden)
            self.ln2 = LayerNorm(self.hidden*2)
            self.ln3 = LayerNorm(self.hidden*2)

        # Leaky Relu activation function
        self.lrelu = nn.LeakyReLU()

        #Output Activation function
        self.tanh = nn.Tanh()

        # Output Activation sigmoid function
        self.sigmoid = nn.Sigmoid()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output_mu.weight)
        nn.init.xavier_uniform_(self.output_logstd.weight)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state, deterministic=False, return_log_prob=True):

        """
                :param state: state
                :param deterministic: If True, do not sample
                :param return_log_prob: If True, return a sample and its log probability
        """

        x = self.input(state)
        if self.use_layernorm:
            x = self.ln1(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        if self.use_layernorm:
            x = self.ln2(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        if self.use_layernorm:
            x = self.ln3(x)
        x = self.lrelu(x)

        mu = self.output_mu(x)
        logstd = self.output_logstd(x)
        # Clamp the log of the standard deviation
        logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(logstd)

        log_prob = None
        pre_sigmoid_value = None

        if deterministic:
            output = nn.Sigmoid()(mu)
        else:
            sigmoid_normal = SigmoidNormal(normal_mean=mu, normal_std=std)
            if return_log_prob:
                output, pre_sigmoid_value = sigmoid_normal.sample(
                    return_pre_sigmoid_value=True
                )
                log_prob = sigmoid_normal.log_prob(
                    output,
                    pre_sigmoid_value=pre_sigmoid_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                output = sigmoid_normal.sample()

        #output = self.reparameterize(mu, logstd)

        if self.use_tanh:
            output = self.tanh(output)

        return output, mu, log_prob, std, logstd, pre_sigmoid_value


# Estimates the Q value of a state action pair
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_dim, output_dim, use_layer_norm=False):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden_dim
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim+self.action_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden*2)
        self.hidden_2 = nn.Linear(in_features=self.hidden*2, out_features=self.hidden*2)
        self.output = nn.Linear(in_features=self.hidden*2, out_features=self.output_dim)

        if self.use_layer_norm:
            self.ln1 = LayerNorm(self.hidden)
            self.ln2 = LayerNorm(self.hidden*2)
            self.ln3 = LayerNorm(self.hidden*2)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state, action):
        #x = torch.cat([state, action], dim=-1)
        x = self.input(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = self.lrelu(x)
        # For the critic, actions are not included until the second hidden layer
        x = torch.cat([x, action], dim=-1)
        x = self.hidden_2(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = self.lrelu(x)
        output = self.output(x)
        return output


# Estimates the value of a state
class ValueNetwork(nn.Module):

    def __init__(self, state_dim,
                 hidden_dim, output_dim):
        super(ValueNetwork, self).__init__()

        self.state_dim = state_dim
        self.hidden = hidden_dim
        self.output_dim = output_dim

        # Architecture
        self.input = nn.Linear(in_features=self.state_dim, out_features=self.hidden)
        self.hidden_1 = nn.Linear(in_features=self.hidden, out_features=self.hidden * 2)
        self.hidden_2 = nn.Linear(in_features=self.hidden * 2, out_features=self.hidden * 2)
        self.output = nn.Linear(in_features=self.hidden * 2, out_features=self.output_dim)

        # Leaky Relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights with xavier initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.hidden_1.weight)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, state):
        x = self.input(state)
        x = self.lrelu(x)
        x = self.hidden_1(x)
        x = self.lrelu(x)
        x = self.hidden_2(x)
        x = self.lrelu(x)
        output = self.output(x)
        return output