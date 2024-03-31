from typing import Optional, Union

import torch

from kooplearn._src.utils import ShapeError
from kooplearn.data import TensorContextDataset


def encode_contexts(contexts_batch: TensorContextDataset, encoder: torch.nn.Module):
    # Caution: this method is designed only for internal calling.
    context_len = contexts_batch.context_length
    batch_size = len(contexts_batch)
    trail_dims = contexts_batch.shape[2:]

    encoded_contexts = encoder(
        contexts_batch.data.view(batch_size * context_len, *trail_dims)
    )
    encoded_contexts = encoded_contexts.view(batch_size, context_len, -1)
    return TensorContextDataset(encoded_contexts)


def decode_contexts(
    encoded_contexts_batch: TensorContextDataset, decoder: torch.nn.Module
):
    # Caution: this method is designed only for internal calling.
    context_len = encoded_contexts_batch.context_length
    batch_size = len(encoded_contexts_batch)
    assert (
        len(encoded_contexts_batch.shape[2:]) == 1
    ), "The decoder input must be a 1-dimensional tensor (i.e. a vector)."

    decoded_contexts = decoder(
        encoded_contexts_batch.data.view(batch_size * context_len, -1)
    )
    trail_dims = decoded_contexts.shape[1:]
    decoded_contexts = decoded_contexts.view(batch_size, context_len, *trail_dims)
    return TensorContextDataset(decoded_contexts)


def evolve_contexts(
    encoded_contexts_batch: TensorContextDataset,
    lookback_len: int,
    forward_operator: Union[torch.nn.Parameter, torch.Tensor],
    backward_operator: Optional[Union[torch.nn.Parameter, torch.Tensor]] = None,
):
    # Caution: this method is designed only for internal calling.
    context_len = encoded_contexts_batch.context_length

    X_init = torch.squeeze(
        encoded_contexts_batch.slice(slice(lookback_len - 1, lookback_len))
    )  # Initial condition  # TODO: Seems like a terrible way to access [:, 0] ??
    if X_init.ndim != 2:
        raise ShapeError(
            f"The encoder network must return a 1D vector for each snapshot, while a shape {X_init.shape[1:]} tensor was received."
        )
    evolved_contexts_batch = torch.zeros_like(encoded_contexts_batch.data)

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
    return TensorContextDataset(
        evolved_contexts_batch
    )  # [batch_size, context_len, latent_dim]


def evolve_forward(
    batch: TensorContextDataset,
    lookback_len: int,
    t: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    forward_operator: Union[torch.nn.Parameter, torch.Tensor],
):
    init_contexts = batch.slice(
        slice(lookback_len - 1, lookback_len)
    )  # Should be of shape [batch_size, 1, *trail_dims]
    init_contexts = init_contexts.view(len(batch), *batch.shape[2:])
    encoded_init_contexts = encoder(init_contexts)
    _raw_encoded_inits = encoded_init_contexts.data.view(len(init_contexts), -1)
    pwd_operator = torch.matrix_power(forward_operator, t)
    _raw_evolved_inits = torch.mm(pwd_operator, _raw_encoded_inits.T).T
    _raw_decoded_preds = decoder(_raw_evolved_inits)
    return TensorContextDataset(
        _raw_decoded_preds.view(len(batch), 1, *batch.shape[2:])
    )


def evolve_batch(
    batch: TensorContextDataset,
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
    batch_evolved = batch_evolved.reshape(batch.shape)
    assert batch.shape == batch_evolved.shape
    return TensorContextDataset(batch_evolved)

def multi_matrix_power(matrix: torch.Tensor, context_length: int, spectral=False):

    if matrix.requires_grad and spectral:
        raise AttributeError("Gradients cannot be reliably computed when using eigendecomposition.")
    if spectral:
        raise NotImplementedError("Spectral decomposition to do matrix powers need to be implemented.")

    matrix_powers = []
    for exp in range(context_length):
        if exp == 0:
            matrix_powers.append(torch.eye(matrix.shape[0], device=matrix.device))
        elif exp == 1:
            matrix_powers.append(matrix)
        else:
            matrix_powers.append(torch.matrix_power(matrix, exp))

    return torch.stack(matrix_powers)


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

def flatten_context_data(x: TensorContextDataset) -> torch.Tensor:
    """ Returns a (batch * time, *feature_dims) view of the context window's data of shape (batch, time, *feature_dims).

    This method guarantees the preservation of temporal sample ordering when reshaping the data.
    It first ensures that the tensor is contiguous in memory, which safeguards against any disruption of data order
    during the reshaping process.

    Returns:
        torch.Tensor: Reshaped view tensor of the data in the shape shape (batch * time, *feature_dims).
    """
    batch_size = len(x)
    context_length = x.context_length

    x.data = x.data.contiguous()  # Needed for reshaping not to mess with the time order, and avoid copying.
    flat_data = x.data.view(batch_size * context_length, *x.shape[2:])
    return flat_data


def unflatten_context_data(x_reshaped: torch.Tensor, batch_size: int, features_shape: tuple) -> TensorContextDataset:
    """ Returns a (batch, context_length, *feature_dims) view of the tensor (batch_size * context_length, *feature_dims).

    Args:
        x_reshaped: Tensor of shape (batch_size * context_length, *feature_dims).
        batch_size: Integer divisor of the first dimension of `x_reshaped`.
        features_shape: Shape of the features.

    Returns:
        TensorContextDataset: Reshaped view tensor of the data in the shape (batch, time, *feature_dims).
    """
    total_elements = x_reshaped.shape[0]
    assert total_elements % batch_size == 0, f"Total elements {total_elements} not divisible by batch size {batch_size}"
    context_length = total_elements // batch_size

    new_shape = (batch_size, context_length) + features_shape
    x_original = x_reshaped.view(new_shape)
    return TensorContextDataset(x_original)