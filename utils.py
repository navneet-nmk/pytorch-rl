import matplotlib.pyplot as plt
import numpy as np
import math

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def soft_update(polyak_factor, target_network, network):
    """
    Soft update of the parameters using Polyak averaging
    :param polyak_factor: The factor by which to move the averages
    :param target_network: The network to load the weights INTO
    :param network: The network to load weights FROM
    """
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))



def hard_update(target_network, network):
    """
    Hard Update of the networks
    :param target_network: The network to load the weights INTO
    :param network: The network to load weights FROM
    """
    target_network.load_state_dict(network.state_dict())


def get_epsilon_iteration(steps_done, EPS_END, EPS_START, EPS_DECAY):
    """
    Used for the epsilon greedy policy (Used in DQN)
    :param steps_done:
    :param EPS_END:
    :param EPS_START:
    :param EPS_DECAY:
    :return:
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold