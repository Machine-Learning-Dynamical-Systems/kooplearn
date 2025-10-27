"""Linear algebra utilities for the `kernel` algorithms."""

from typing import Optional
from warnings import warn

import numpy as np
from numpy import ndarray

from kooplearn._utils import stable_topk


def add_diagonal_(M: ndarray, alpha: float):
    """Add alpha to the diagonal of M inplace.

    Args:
        M (ndarray): The matrix to modify inplace.
        alpha (float): The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def spd_neg_pow(
    M: np.ndarray,
    exponent: float = -1.0,
    cutoff: Optional[float] = None,
    strategy: str = "trunc",
) -> np.ndarray:
    """
    Truncated eigenvalue decomposition of A
    """
    if cutoff is None:
        cutoff = 10.0 * M.shape[0] * np.finfo(M.dtype).eps
    w, v = np.linalg.eigh(M)
    if strategy == "trunc":
        sanitized_w = np.where(w <= cutoff, 1.0, w)
        inv_w = np.where(
            w > cutoff, (sanitized_w ** np.abs(exponent)) ** np.sign(exponent), 0.0
        )
        v = np.where(w > cutoff, v, 0.0)
    elif strategy == "tikhonov":
        inv_w = ((w + cutoff) ** np.abs(exponent)) ** np.sign(exponent)
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")
    return np.linalg.multi_dot([v, np.diag(inv_w), v.T])


def covariance(X: np.ndarray, Y: Optional[np.ndarray] = None):
    X = np.atleast_2d(X)
    if X.ndim > 2:
        raise ValueError(f"Input array has more than 2 dimensions ({X.ndim}).")
    rnorm = (X.shape[0]) ** (-0.5)
    X = X * rnorm

    if Y is None:
        c = X.T @ X
    else:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Shape mismatch: the covariance between two arrays can be computed only if they have the same initial dimension. Got {X.shape[0]} and {Y.shape[0]}."
            )
        Y = np.atleast_2d(Y)
        if Y.ndim > 2:
            raise ValueError(f"Input array has more than 2 dimensions ({Y.ndim}).")
        Y = Y * rnorm
        c = X.T @ Y
    return c


def weighted_norm(A: ndarray, M: ndarray | None = None):
    r"""Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector :math:`a` is given by :math:`\langle a, Ma \rangle` . Defaults to None, corresponding to the Identity matrix. Warning: no checks are
        performed on M being a PSD operator.

    Returns:
        (ndarray or float): If ``A.ndim == 2`` returns 1D array of floats corresponding to the norms of
        the columns of A. Else return a float.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    if M is None:
        norm = np.linalg.norm(A, axis=0)
    else:
        _A = np.dot(M, A)
        _A_T = np.dot(M.T, A)
        norm = np.real(np.sum(0.5 * (np.conj(A) * _A + np.conj(A) * _A_T), axis=0))
    rcond = 10.0 * A.shape[0] * np.finfo(A.dtype).eps
    norm = np.where(norm < rcond, 0.0, norm)
    return np.sqrt(norm)


def eigh_rank_reveal(
    values: np.ndarray,
    vectors: np.ndarray,
    rank: int,  # Desired rank
    rcond: Optional[float] = None,  # Threshold for the singular values
    ignore_warnings: bool = True,
):
    if rcond is None:
        rcond = 10.0 * values.shape[0] * np.finfo(values.dtype).eps
    top_vals, indices = stable_topk(values, rank)
    vectors = vectors[:, indices]
    values = top_vals

    _ftest = values > rcond
    if all(_ftest):
        rsqrt_vals = (np.sqrt(values)) ** -1
    else:
        first_invalid = np.argmax(
            ~_ftest
        )  # In the case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        _first_discarded_val = np.max(np.abs(values[first_invalid:]))
        values = values[_ftest]
        vectors = vectors[:, _ftest]

        if not ignore_warnings:
            warn(
                f"Warning: Discarted {rank - vectors.shape[1]} dimensions of the {rank} requested due to numerical instability. Consider decreasing the rank. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        # Compute stable sqrt
        rsqrt_vals = (np.sqrt(values)) ** -1
    return vectors, values, rsqrt_vals
