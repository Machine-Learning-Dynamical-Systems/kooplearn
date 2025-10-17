"""Linear algebra utilities for the `kernel` algorithms."""

from warnings import warn
from typing import Optional

import numpy as np
from numpy import ndarray

from kooplearn.utils import topk


def add_diagonal_(M: ndarray, alpha: float):
    """Add alpha to the diagonal of M inplace.

    Args:
        M (ndarray): The matrix to modify inplace.
        alpha (float): The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def stable_topk(
    vec: ndarray,
    k_max: int,
    rcond: float | None = None,
    ignore_warnings: bool = True,
):
    """Takes up to k_max indices of the top k_max values of vec. If the values are below rcond, they are discarded.

    Args:
        vec (ndarray): Vector to extract the top k indices from.
        k_max (int): Number of indices to extract.
        rcond (float, optional): Value below which the values are discarded. Defaults to None, in which case it is set according to the machine precision of vec's dtype.
        ignore_warnings (bool): If False, raise a warning when some elements are discarted for being below the requested numerical precision.

    """

    if rcond is None:
        rcond = 10.0 * vec.shape[0] * np.finfo(vec.dtype).eps

    top_vec, top_idxs = topk(vec, k_max)

    if all(top_vec > rcond):
        return top_vec, top_idxs
    else:
        valid = top_vec > rcond
        # In the case of multiple occurrences of the maximum vec, the indices corresponding to the first occurrence are returned.
        first_invalid = np.argmax(np.logical_not(valid))
        _first_discarded_val = np.max(np.abs(vec[first_invalid:]))

        if not ignore_warnings:
            warn(
                f"Warning: Discarted {k_max - np.sum(valid)} dimensions of the {k_max} requested due to numerical instability. Consider decreasing the k. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        return top_vec[valid], top_idxs[valid]


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
    top_vals, indices = topk(values, rank)
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