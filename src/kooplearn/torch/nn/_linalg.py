"""Linear Algebra."""

from typing import NamedTuple

import torch
from torch import Tensor


def sqrtmh(A: Tensor) -> Tensor:
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices.

    Used code from `this issue <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_.

    Args:
        A (Tensor): Symmetric or Hermitian positive definite matrix or batch of matrices.

    Shape:
        ``A``: :math:`(N, N)`

        Output: :math:`(N, N)`
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


####################################################################################################
# TODO: THIS IS JUST COPY AND PASTE FROM OLD NCP
# Should topk and filter_reduced_rank_svals be in utils? They look like linalg to me, specially the
# filter
####################################################################################################


# Sorting and parsing
class TopKReturnType(NamedTuple):  # noqa: D101
    values: Tensor
    indices: Tensor


def topk(vec: Tensor, k: int):  # noqa: D103
    assert vec.ndim == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = torch.flip(torch.argsort(vec), dims=[0])  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return TopKReturnType(values, indices)


def filter_reduced_rank_svals(values, vectors):  # noqa: D103
    eps = 2 * torch.finfo(torch.get_default_dtype()).eps
    # Filtering procedure.
    # Create a mask which is True when the real part of the eigenvalue is negative or the imaginary part is nonzero
    is_invalid = torch.logical_or(
        torch.real(values) <= eps,
        torch.imag(values) != 0
        if torch.is_complex(values)
        else torch.zeros(len(values), device=values.device),
    )
    # Check if any is invalid take the first occurrence of a True value in the mask and filter everything after that
    if torch.any(is_invalid):
        values = values[~is_invalid].real
        vectors = vectors[:, ~is_invalid]

    sort_perm = topk(values, len(values)).indices
    values = values[sort_perm]
    vectors = vectors[:, sort_perm]

    # Assert that the eigenvectors do not have any imaginary part
    assert torch.all(
        torch.imag(vectors) == 0 if torch.is_complex(values) else torch.ones(len(values))
    ), "The eigenvectors should be real. Decrease the rank or increase the regularization strength."

    # Take the real part of the eigenvectors
    vectors = torch.real(vectors)
    values = torch.real(values)
    return values, vectors
