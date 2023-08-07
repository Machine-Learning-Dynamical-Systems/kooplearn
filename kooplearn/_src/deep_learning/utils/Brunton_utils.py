import torch
import torch.nn as nn
from copy import deepcopy


class AuxiliaryNetworkWrapper(nn.Module):
    # This module should work with any architecture that we want as long as it takes the input_dim and output_dim as
    # arguments. However, note that for partially connected layers (as originally suggested in the paper) this is not
    # the most efficient implementation. We could for example stack multiple non-fully connected layers using a
    # grouped 1d convolution (as suggested in
    # https://stackoverflow.com/questions/70269663/how-to-efficiently-implement-a-non-fully-connected-linear-layer-in
    # -pytorch), I believe that this is also the idea of a depthwise convolution (
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and (
    # https://www.youtube.com/watch?v=vVaRhZXovbw). Anyway, the original implementation also uses a for loop,
    # so it should be feasible.
    def __init__(self, model_architecture, model_hyperparameters, num_complex_pairs, num_real):
        super().__init__()
        model_complex_pairs = model_architecture(**model_hyperparameters, input_dim=1, output_dim=2)
        model_real = model_architecture(**model_hyperparameters, input_dim=1, output_dim=1)
        self.models_complex_pairs = nn.ModuleList([deepcopy(model_complex_pairs) for _ in range(num_complex_pairs)])
        self.models_real = nn.ModuleList([deepcopy(model_real) for _ in range(num_real)])

    def forward(self, y):
        # dimensions convention (..., dimension_of_autoencoder_subspace)
        # note that the dimension_of_autoencoder_subspace must be equal to 2 * num_complex_pairs + num_real
        radius_of_pairs = y[..., 0:-len(self.models_real):2]**2 + y[..., 1:-len(self.models_real):2]**2
        mus_omegas = []
        lambdas = []
        for i, model in enumerate(self.models_complex_pairs):
            mus_omegas.append(model(radius_of_pairs[..., i]))
        for i, model in enumerate(self.models_real):
            lambdas.append(model(y[..., -len(self.models_real) + i]))
        return torch.cat(mus_omegas, dim=-1), torch.cat(lambdas, dim=-1)


def advance_encoder_output(encoded_x_value, mus_omegas, lambdas):
    # we will try to eliminate the for loop of the original implementation....must check if it works
    cos_omega = torch.cos(mus_omegas[..., 1])
    sin_omega = torch.sin(mus_omegas[..., 1])
    jordan_block = torch.stack([torch.cat([cos_omega, -sin_omega], dim=-1),
                                torch.cat([sin_omega, cos_omega], dim=-1)],
                               dim=-1)  # should be of shape (..., 2, 2)
    jordan_block = torch.exp(mus_omegas[..., 0]) * jordan_block
    y_complex_pairs_matrix = torch.stack(
        [torch.cat([encoded_x_value[..., 0:-len(lambdas):2], encoded_x_value[..., 1:-len(lambdas):2]], dim=-1),
         torch.cat([encoded_x_value[..., 0:-len(lambdas):2], encoded_x_value[..., 1:-len(lambdas):2]], dim=-1)],
        dim=-1)  # should be of shape (..., num_complex_pairs, 2, 2)
    # next should be of shape (..., num_complex_pairs)
    y_complex_pairs = torch.einsum('...ij,...kij -> ...k', jordan_block, y_complex_pairs_matrix)
    y_real = encoded_x_value[..., -len(lambdas):]*torch.exp(lambdas)
    return torch.cat([y_complex_pairs, y_real], dim=-1)

