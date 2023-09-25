from typing import Optional, Union

import torch

from kooplearn._src.utils import ShapeError


# A bit of code copy paste but it's ok for now
def encode_contexts(contexts_batch: torch.Tensor, encoder: torch.nn.Module):
    # Caution: this method is designed only for internal calling.
    context_len = contexts_batch.shape[1]
    Z = []
    for i in range(context_len):  # Inefficient but working
        X = contexts_batch[:, i, ...]
        Z.append(encoder(X))
    Z = torch.stack(Z, dim=1)
    latent_dim = Z.shape[2:]
    assert (
        len(latent_dim) == 1
    ), "The encoder output must be a 1-dimensional tensor (i.e. a vector)."
    return Z


def decode_contexts(encoded_contexts_batch: torch.Tensor, decoder: torch.nn.Module):
    # Caution: this method is designed only for internal calling.
    context_len = encoded_contexts_batch.shape[1]
    assert (
        len(encoded_contexts_batch.shape[2:]) == 1
    ), "The decoder input must be a 1-dimensional tensor (i.e. a vector)."
    Z = []
    for i in range(context_len):  # Inefficient but working
        X = encoded_contexts_batch[:, i, ...]
        Z.append(decoder(X))
    Z = torch.stack(Z, dim=1)
    return Z


def evolve_contexts(
    encoded_contexts_batch: torch.Tensor,
    lookback_len: int,
    forward_operator: Union[torch.nn.Parameter, torch.Tensor],
    backward_operator: Optional[Union[torch.nn.Parameter, torch.Tensor]] = None,
):
    # Caution: this method is designed only for internal calling.
    context_len = encoded_contexts_batch.shape[1]
    X_init = encoded_contexts_batch[:, lookback_len - 1, ...]  # Initial condition
    if X_init.ndim != 2:
        raise ShapeError(
            f"The encoder network must return a 1D vector for each snapshot, while a shape {X_init.shape[1:]} tensor was received."
        )
    evolved_contexts_batch = torch.zeros_like(encoded_contexts_batch)

    for ctx_idx in range(context_len):  # Not efficient but working
        exp = ctx_idx - lookback_len + 1
        if exp < 0:
            if backward_operator is None:
                raise ValueError(
                    "No backward operator provided, but lookback length greater than 1."
                )
            else:
                pwd_operator = torch.matrix_power(backward_operator, -exp)
                Z = torch.mm(pwd_operator, X_init.T).T
        else:
            pwd_operator = torch.matrix_power(forward_operator, exp)
            Z = torch.mm(pwd_operator, X_init.T).T
        evolved_contexts_batch[:, ctx_idx, ...] = Z
    return evolved_contexts_batch  # [batch_size, context_len, latent_dim]


def evolve_batch(
    batch: torch.Tensor,
    t: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    forward_operator: Union[torch.nn.Parameter, torch.Tensor],
):
    Z = encoder(batch)
    assert (
        len(Z.shape[1:]) == 1
    ), "The encoder output must be a 1-dimensional tensor (i.e. a vector)."
    evolution_operator = torch.matrix_power(forward_operator, t)
    Z_evolved = torch.mm(evolution_operator, Z.T).T
    batch_evolved = decoder(Z_evolved)
    assert batch.shape == batch_evolved.shape
    return batch_evolved


def consistency_loss(
    forward_operator: Union[torch.nn.Parameter, torch.Tensor],
    backward_operator: Union[torch.nn.Parameter, torch.Tensor],
):
    assert (
        forward_operator.shape == backward_operator.shape
    ), "The forward and backward operators must have the same shape."
    terms = []
    dim = forward_operator.shape[0]
    for k in range(1, dim + 1):
        fb = torch.mm(forward_operator[:k, :], backward_operator[:, :k]) - torch.eye(
            k, device=forward_operator.device
        )
        bf = torch.mm(backward_operator[:k, :], forward_operator[:, :k]) - torch.eye(
            k, device=forward_operator.device
        )
        terms.append(
            torch.linalg.matrix_norm(fb) ** 2 + torch.linalg.matrix_norm(bf) ** 2
        )
    return 0.5 * torch.mean(torch.stack(terms))
