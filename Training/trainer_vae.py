"""
Class for a generic trainer used for training all the different generative models
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

    def __init__(self,
                 generative_model, learning_rate, num_epochs,
                 random_seed, output_folder, multi_gpu_training=False,
                 use_cuda=True, save_model=True, verbose=True,
                 plot_stats=True):

        """

        :param generative_model: The generative model (eg VAE) to train
        :param learning_rate: learning rate for the optimizer
        :param num_epochs: total number of training steps
        :param random_seed: set the random seed for reproduction of the results
        :param output_folder: output folder for the saved model
        :param use_cuda: use cuda in case of availability of gpu
        :param save_model: save the generative model weights
        :param verbose: print the training statements
        :param plot_stats: plot the stats of training
        """

        self.model = generative_model
        self.lr = learning_rate
        self.seed = random_seed
        self.num_epochs = num_epochs
        