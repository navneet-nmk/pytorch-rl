"""

This script contains the implementation of the parameter perturbations as suggested in
PARAMETER SPACE NOISE FOR EXPLORATION, Plappert et al.

"""

import torch


class ParameterNoise(object):

    def __init__(self, actor, perturbed_actor, param_noise_stddev):

        self.actor = actor
        self.perturbed_actor = perturbed_actor
        self.param_noise_stddev = param_noise_stddev

    def get_perturbable_parameters(self, model):
        parameters = []
        for name, params in model.named_parameters():
            if 'ln' not in name:
                parameters.append(params)

        return parameters

    def get_perturbed_actor_updates(self):
        """

        Update the perturbed actor parameters

        :return:
        """
        assert len(self.actor.parameters()) == len(self.perturbed_actor.parameters())
        actor_perturbable_parameters = self.get_perturbable_parameters(self.actor)
        perturbed_actor_perturbable_parameters = self.get_perturbable_parameters(self.perturbed_actor)
        assert len(actor_perturbable_parameters) == len(perturbed_actor_perturbable_parameters)

        for params, perturbed_params in zip(actor_perturbable_parameters, perturbed_actor_perturbable_parameters):
            # Update the parameters
            perturbed_params.data.copy_(params + torch.normal(mean=torch.zeros(params.shape),
                                                              std=self.param_noise_stddev))









