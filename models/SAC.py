"""

This script contains an implementation of the Soft Actor Critic.

This is an off policy Actor critic algorithm with the entropy of
the current policy added to the reward.

The maximization of the augmented reward enables the agent to discover multimodal actions
(Multiple actions that results in reward). It promotes exploration
and is consoderably more stable to random seeds variation and hyperparameter
tuning compared to Deep Deterministic Policy Gradient.

"""

import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

class SAC(nn.Module):

    def __init__(self):
        super(SAC, self).__init__()




class StochasticActor(nn.Module):

    def __init__(self):
        super(StochasticActor, self).__init__()


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        