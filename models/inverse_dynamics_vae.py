"""

This script contains an implementation of the inverse dynamics variational autoencoder.
This basically combines the inverse dynamics model (predicting the action from the
current and the next state) and the variational autoencoder trained with reconstruction error.
As an added parameter, beta is included with the KL divergence inorder to encourage
disentangled representations.

"""

import torch
import torch.nn as nn
from torch.autograd import Variable


# The encoder for the INVAE
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()


# The decoder for the INVAE
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        
# The inverse dynamic module
class InverseDM(nn.Module):

    def __init__(self):
        super(InverseDM, self).__init__()


class INVAE(nn.Module):

    def __init__(self):
        super(INVAE, self).__init__()