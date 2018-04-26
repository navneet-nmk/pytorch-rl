import torch

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, use_cuda):
        """
        A storage class for storing the episode rollouts across various environments
        :param num_steps:
        :param num_processes:
        :param obs_shape:
        :param action_space:
        :param state_size:
        :param use_cuda:
        """
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        action_shape = 1

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

    def insert(self, step, current_obs, state, action, action_log_prob, value_pred, reward, mask):
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
        """

        :return:
        """
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])



    def compute_returns(self, next_value, use_gae, gamma, tau):
        """
        This function is being used to compute the true state values using a bootstrapped
        estimate and backtracking.

        :param next_value:
        :param use_gae:
        :param gamma:
        :param tau:
        :return:
        """
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step]