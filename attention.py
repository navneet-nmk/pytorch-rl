"""
This file contains the self attention and the multi attention modules
"""
import torch
import torch.nn as nn


class MultiAttention(nn.Module):
    """
    The Mutli Attention module
    """
    def __init__(self):
        super(MultiAttention, self).__init__()


class SelfAttention(nn.Module):
    """
    The Self Attention module
    """
    def __init__(self):
        super(SelfAttention, self).__init__()


class GoalNetwork(nn.Module):
    """
    This network uses the self and multi attention modules and returns the
    top n vectors according to the softmax probabilities.
    """

    def __init__(self):
        super(GoalNetwork, self).__init__()