"""

This script contains the implementation of the VAE-GAN model
that combines a vartiational autoencoder and a GAN.

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable