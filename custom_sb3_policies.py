import gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            num_inputs_vector,
            num_inputs_nodes,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():

            if key == "nodes":
                extractors[key] = nn.Linear(num_inputs_nodes, 64)
                total_concat_size += 64

            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(num_inputs_vector, 64)
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        self._nodes = np.array([])

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        if (self._nodes != observations['nodes']).any():
            self._encoded_nodes = self.extractors['nodes'](observations['nodes'])
            self._nodes = observations['nodes']

        encoded_tensor_list.append(self._encoded_nodes)
        encoded_tensor_list.append(self.extractors['vector'](observations['vector']))

        return th.cat(encoded_tensor_list, dim=1)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)