from typing import Type

import torch
import torch.nn as nn
from copy import deepcopy


class AuxiliaryNetworkWrapper(nn.Module):
    """Wrapper for the auxiliary network used in [1].

    This wrapper allows us to use any architecture for the auxiliary network as long as it takes the input_dim and
    output_dim as arguments.

    Parameters:

        model_architecture: Architecture of the auxiliary network. Can be any deep learning architecture
            (torch.nn.Module) that will be wrapped in a AuxiliaryNetworkWrapper. The auxiliary
            network must take as input a dictionary containing the key 'x_value', a tensor of shape (..., input_dim) and
            outputs a tensor of shape (..., output_dim) the auxiliary_network_class must take the keyword argument
            input_dim and output_dim when being instantiated, which will be correctly set by this wrapper.
        model_hyperparameters: Hyperparameters of the auxiliary network. Must be a dictionary containing as
            keys the names of the hyperparameters and as values the values of the hyperparameters of the auxiliary
            network. Note that the keyword arguments input_dim and output_dim will be set by this wrapper, so they
            should not be included in model_hyperparameters.
        num_complex_pairs: Number of complex pairs parametrized by the auxiliary network.
        num_real: Number of real numbers parametrized by the auxiliary network.

    [1] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. “Deep Learning for Universal Linear Embeddings of
    Nonlinear Dynamics.” Nature Communications 9, no. 1 (November 23, 2018): 4950.
    https://doi.org/10.1038/s41467-018-07210-0.
    """

    # This module should work with any architecture that we want as long as it takes the input_dim and output_dim as
    # arguments. However, note that for partially connected layers (as originally suggested in the paper) this is not
    # the most efficient implementation. We could for example stack multiple non-fully connected layers using a
    # grouped 1d convolution (as suggested in
    # https://stackoverflow.com/questions/70269663/how-to-efficiently-implement-a-non-fully-connected-linear-layer-in
    # -pytorch), I believe that this is also the idea of a depthwise convolution (
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and (
    # https://www.youtube.com/watch?v=vVaRhZXovbw). Anyway, the original implementation also uses a for loop,
    # so it should be feasible.
    def __init__(self, model_architecture: Type[nn.Module], model_hyperparameters: dict,
                 num_complex_pairs: int, num_real: int):
        super().__init__()
        model_complex_pairs = model_architecture(**model_hyperparameters, input_dim=1, output_dim=2)
        model_real = model_architecture(**model_hyperparameters, input_dim=1, output_dim=1)
        self.models_complex_pairs = nn.ModuleList([deepcopy(model_complex_pairs) for _ in range(num_complex_pairs)])
        self.models_real = nn.ModuleList([deepcopy(model_real) for _ in range(num_real)])

    def forward(self, y):
        # dimensions convention (..., dimension_of_autoencoder_subspace)
        # note that the dimension_of_autoencoder_subspace must be equal to 2 * num_complex_pairs + num_real
        radius_of_pairs = y[..., 0:-len(self.models_real):2] ** 2 + y[..., 1:-len(self.models_real):2] ** 2
        mus_omegas = []
        lambdas = []
        for i, model in enumerate(self.models_complex_pairs):
            mus_omegas.append(model({'x_value': radius_of_pairs[..., i][..., None]}))
        for i, model in enumerate(self.models_real):
            lambdas.append(model({'x_value': y[..., -len(self.models_real) + i][..., None]}))
        return torch.stack(mus_omegas, dim=-2), torch.cat(lambdas, dim=-1)


def advance_encoder_output(encoded_x_value, mus_omegas, lambdas):
    # we will try to eliminate the for loop of the original implementation....must check if it works
    cos_omega = torch.cos(mus_omegas[..., 1])
    sin_omega = torch.sin(mus_omegas[..., 1])
    jordan_block = torch.stack([torch.stack([cos_omega, -sin_omega], dim=-1),
                                torch.stack([sin_omega, cos_omega], dim=-1)],
                               dim=-1)  # should be of shape (..., num_complex_pairs, 2, 2)
    jordan_block = torch.exp(mus_omegas[..., 0])[..., None, None] * jordan_block
    y_complex_pairs_matrix = torch.stack(
        [
            torch.stack([encoded_x_value[..., 0:-lambdas.shape[-1]:2], encoded_x_value[..., 1:-lambdas.shape[-1]:2]],
                        dim=-1),
            torch.stack([encoded_x_value[..., 0:-lambdas.shape[-1]:2], encoded_x_value[..., 1:-lambdas.shape[-1]:2]],
                        dim=-1)],
        dim=-1)  # should be of shape (..., num_complex_pairs, 2, 2)
    # next should be of shape (..., 2*num_complex_pairs)
    y_complex_pairs = (jordan_block * y_complex_pairs_matrix).sum(dim=-2).flatten(start_dim=-2, end_dim=-1)
    y_real = encoded_x_value[..., -lambdas.shape[-1]:] * torch.exp(lambdas)
    return torch.cat([y_complex_pairs, y_real], dim=-1)
