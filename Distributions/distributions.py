import torch
import torch.nn.functional as F

try:
    from torch.distributions import Distribution, Normal
except ImportError:
    print("You should use a PyTorch version that has torch.distributions.")

    import math
    from numbers import Number

    # Base Class distribution
    class Distribution(object):
        r"""
        Distribution is the abstract base class for probability distributions.
        """

        def sample(self):
            """
            Generates a single sample or single batch of samples if the distribution
            parameters are batched.
            """
            raise NotImplementedError

        def sample_n(self, n):
            """
            Generates n samples or n batches of samples if the distribution parameters
            are batched.
            """
            raise NotImplementedError

        def log_prob(self, value):
            """
            Returns the log of the probability density/mass function evaluated at
            `value`.
            Args:
                value (Tensor or Variable):
            """
            raise NotImplementedError


# Sigmoid Normal Distribution
class SigmoidNormal(Distribution):
    """
        Represent distribution of X where
            X ~ sigmoid(Z)
            Z ~ N(mean, std)
        Note: this is not very numerically stable.
        """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        super(SigmoidNormal, self).__init__()
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_sigmoid_value=False):
        z = self.normal.sample_n(n)
        if return_pre_sigmoid_value:
            return F.sigmoid(z), z
        else:
            return F.sigmoid(z)

    def log_prob(self, value, pre_sigmoid_value=None):
        """
        :param value: some value, x
        :param pre_sigmoid_value: arcsigmoid(x)
        :return:
        """
        if pre_sigmoid_value is None:
            pre_sigmoid_value = torch.log(
                    (value) / (1 - value)
                )
        return self.normal.log_prob(pre_sigmoid_value) - torch.log(
                value*(1-value) + self.epsilon
        )

    def sample(self, return_pre_sigmoid_value=False):
        z = self.normal.sample()
        if return_pre_sigmoid_value:
            return F.sigmoid(z), z
        else:
            return F.sigmoid(z)


# Tanh Normal Distribution
class TanhNormal(Distribution):
    """
        Represent distribution of X where
            X ~ tanh(Z)
            Z ~ N(mean, std)
        Note: this is not very numerically stable.
        """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        super(TanhNormal, self).__init__()
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return F.tanh(z), z
        else:
            return F.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_sigmoid_value: arcsigmoid(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_sigmoid_value = torch.log(
                    (1+value) / (1 - value)
                )/2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
                1- value*value + self.epsilon
        )

    def sample(self, return_pre_tanh_value=False):
        z = self.normal.sample()
        if return_pre_tanh_value:
            return F.tanh(z), z
        else:
            return F.tanh(z)


