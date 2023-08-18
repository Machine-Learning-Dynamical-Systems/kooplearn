from typing import Callable

import torch.nn as nn


class MLPModel(nn.Module):
    """Simple MLP model.

    Parameters:
        input_dim: input dimension
        output_dim: output dimension
        hidden_dims: list of hidden dimensions
        flatten_input: if True, flatten the input before feeding it to the MLP
        output_activation_fn: activation function for the output layer
        activation_fn: activation function for the hidden layers
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], flatten_input: bool = True,
                 output_activation_fn: Callable = nn.Identity(),
                 activation_fn: Callable = nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.output_activation_fn = output_activation_fn
        self.activation_fn = activation_fn
        self.flatten_input = flatten_input
        layer_dims = [input_dim] + hidden_dims
        if flatten_input:
            module_list = [nn.Flatten(start_dim=-2, end_dim=-1)]
        else:
            module_list = []
        for i, dim in enumerate(layer_dims[:-1]):
            module_list.extend([nn.Linear(dim, layer_dims[i + 1]), activation_fn])
        module_list.extend([nn.Linear(layer_dims[-1], output_dim), output_activation_fn])
        self.sequential = nn.Sequential(*module_list)

    def forward(self, data):
        """Forward pass.

        Args:
            data: dictionary with key 'x_value' containing the input data

        Returns:
            output of the MLP
        """
        # dimensions convention (..., channels, temporal_dim)
        x = data['x_value']
        return self.sequential(x)
