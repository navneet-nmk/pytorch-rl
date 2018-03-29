import numpy as np


# Base Random Process Class
class RandomProcess(object):

    def reset_states(self):
        pass



# Annealed Gaussian
class AnnealedGaussianProcess(RandomProcess):

    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n = 0


        if sigma_min is not None:
            self.m = - float(sigma - sigma_min)/ float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m*float(self.n) + self.c)
        return sigma



class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, )