import torch
USE_CUDA = torch.cuda.is_available()

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape,
                 action_space, state_size, use_cuda, action_shape):
        """
        A storage class for storing the episode rollouts across various environments
        :param num_steps: Steps into the environment
        :param num_processes: Parallel workers collecting the experiences
        :param obs_shape: Shape of the observation
        :param action_space: Action Shape
        :param state_size: Shape of the state (Maybe similar to the observation)
        :param use_cuda: Use GPU
        """
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.action_shape = action_shape
        self.state_size = state_size
        self.use_cuda = use_cuda

        action_shape = self.action_shape

        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step,
               current_obs, state, action, action_log_prob, value_pred, reward, mask):
        """

        :param step:
        :param current_obs:
        :param state:
        :param action:
        :param action_log_prob:
        :param value_pred:
        :param reward:
        :param mask:
        :return:
        """
        self.observations[step + 1].copy_(current_obs)
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        """
        This function is being used to compute the true state values using a bootstrapped
        estimate and backtracking.

        :param next_value:
        :param use_gae: Use generalized advantage estimation
        :param gamma: Discount factor
        :param tau:
        :return:
        """
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step]


    def feed_forward_generator(self, advantages, batch_size):
        pass