"""

This script contains an implementation of the VQ-VAE ie
Vector Quantised Variational Autoencoder.

"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

class VQ_VAE(nn.Module):

    def __init__(self):
        super(VQ_VAE, self).__init__()
