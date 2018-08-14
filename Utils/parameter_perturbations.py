"""

This script contains the implementation of the parameter perturbations as suggested in
PARAMETER SPACE NOISE FOR EXPLORATION, Plappert et al.

"""

import torch
from copy import copy


class ParameterNoise(object):

    def __init__(self, actor, param_noise_stddev,
                 param_noise=None,
                 normalized_observation=None):

        self.actor = actor
        self.param_noise_stddev = param_noise_stddev
        self.param_noise = param_noise
        self.normalized_observation= normalized_observation

    def get_perturbable_parameters(self, model):
        # Removing parameters that don't require parameter noise
        parameters = []
        for name, params in model.named_parameters():
            if 'ln' not in name:
                parameters.append(params)

        return parameters

    def set_perturbed_actor_updates(self, model):
        """

        Update the perturbed actor parameters

        :return:
        """
        assert len(self.actor.parameters()) == len(model.parameters())
        actor_perturbable_parameters = self.get_perturbable_parameters(self.actor)
        perturbed_actor_perturbable_parameters = self.get_perturbable_parameters(model)
        assert len(actor_perturbable_parameters) == len(perturbed_actor_perturbable_parameters)

        for params, perturbed_params in zip(actor_perturbable_parameters, perturbed_actor_perturbable_parameters):
            # Update the parameters
            perturbed_params.data.copy_(params + torch.normal(mean=torch.zeros(params.shape),
                                                              std=self.param_noise_stddev))

    def setup_param_noise(self, normalized_observation):

        assert self.param_noise_stddev is not None
        # Configure perturbed actor
        self.perturbed_actor = copy(self.actor)
        # Perturb the perturbed actor weights
        self.set_perturbed_actor_updates(self.perturbed_actor)
        # Configure separate copy for stddev adoption
        self.adaptive_perturbed_actor = copy(self.actor)
        # Perturb the adaptive actor weights
        self.set_perturbed_actor_updates(self.adaptive_perturbed_actor)
        # Refer to https://arxiv.org/pdf/1706.01905.pdf for details on the distance used specifically for DDPG
        self.adaptive_policy_distance = torch.pow(torch.mean(torch.pow(self.actor(normalized_observation) -
                                                                       self.adaptive_perturbed_actor(normalized_observation)
                                                                       , 2)), 0.5)

        return self.adaptive_policy_distance

    def adapt_param_noise(self, state):
        if self.param_noise_stddev is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        self.set_perturbed_actor_updates(self.perturbed_actor)
        adaptive_noise_distance = self.setup_param_noise(state)
        self.param_noise.adapt(adaptive_noise_distance)

    def reset(self):
        # Reset internal state after an episode is complete
        if self.param_noise is not None:
            self.param_noise_stddev = self.param_noise.current_stddev
            self.set_perturbed_actor_updates(self.perturbed_actor)