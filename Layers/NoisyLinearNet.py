"""

This script contains the implementation of the NoisyNets paper.
This is a Noisy Linear Net which aids exploration by learning perturbations
to the weights of the individual layers.

"""

import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

class NoisyLinearLayer(nn.Module):

    def __init__(self):
        super(NoisyLinearLayer, self).__init__()
        