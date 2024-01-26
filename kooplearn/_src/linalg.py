import logging
from typing import Optional
from warnings import warn

import numpy as np
from sklearn.utils import check_array

from kooplearn._src.utils import topk

logger = logging.getLogger("kooplearn")


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


def weighted_norm(A: np.ndarray, M: Optional[np.ndarray] = None):
    r"""Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector :math:`a` is given by
        :math:`\langle a, Ma\rangle`. Defaults to None, corresponding to the Identity matrix. Warning: no checks are
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


def weighted_dot_product(A: np.ndarray, B: np.ndarray, M: Optional[np.ndarray] = None):
    """Weighted dot product between the columns of A and B. The output will be equivalent to :math:`A^{*} M B`
    if A and B are 2D arrays.

    Args:
        A, B (ndarray): 1D or 2D arrays.
        M (ndarray or LinearOperator, optional): Weigthing matrix. Defaults to None, corresponding to the
        Identity matrix. Warning: no checks are performed on M being a PSD operator.

    Returns:
        (ndarray or float): The result of :math:`A^{*} M B`.
    """
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    assert B.ndim <= 2, "'B' must be a vector or a 2D array"
    A_adj = np.conj(A.T)
    if M is None:
        return np.dot(A_adj, B)
    else:
        _B = np.dot(M, B)
        return np.dot(A_adj, _B)


def _column_pivot(Q, R, k, squared_norms, columns_permutation):
    """
    Helper function to perform column pivoting on the QR decomposition at the k iteration. No checks are performed.
    For internal use only.
    """
    _arg_max = np.argmax(squared_norms[k:])
    j = k + _arg_max
    _in = [k, j]
    _swap = [j, k]
    # Column pivoting
    columns_permutation[_in] = columns_permutation[_swap]
    Q[:, _in] = Q[:, _swap]
    R[:k, _in] = R[:k, _swap]
    squared_norms[_in] = squared_norms[_swap]
    return Q, R, squared_norms, columns_permutation


def modified_QR(
    A: np.ndarray,
    M: Optional[np.ndarray] = None,
    column_pivoting: bool = False,
    rtol: Optional[float] = None,
    verbose: bool = False,
):
    """Modified QR algorithm with column pivoting. Implementation follows the algorithm described in [1].

    Args:
        A (ndarray): 2D array whose columns are vectors to be orthogonalized.
        M (ndarray or LinearOperator, optional): PSD linear operator. If not None, the vectors are orthonormalized with
         respect to the scalar product induced by M. Defaults to None corresponding to Identity matrix.
        column_pivoting (bool, optional): Whether column pivoting is performed. Defaults to False.
        rtol (float, optional): relative tolerance in determining the numerical rank of A. Defaults to 10*A.shape[0]*eps.
        This parameter is used only when ``column_pivoting == True``.
        verbose (bool, optional): Whether to print informations and warnings about the progress of the algorithm.
        Defaults to False.

    Returns:
        tuple: A tuple of the form (Q, R), where Q and R satisfy A = QR. If ``column_pivoting == True``, the permutation
         of the columns of A is returned as well.

    [1] A. Dax: 'A modified Gram–Schmidt algorithm with iterative orthogonalization and column pivoting',
    https://doi.org/10.1016/S0024-3795(00)00022-7.
    """
    A = check_array(A)  # Ensure A is non-empty 2D array containing only finite values.
    num_vecs = A.shape[1]
    effective_rank = num_vecs
    dtype = A.dtype
    Q = np.copy(A)
    R = np.zeros((num_vecs, num_vecs), dtype=dtype)

    if rtol is None:
        rtol = 10.0 * A.shape[0] * np.finfo(A.dtype).eps
    _roundoff = 1e-8  # From reference paper
    _tau = 1e-2  # From reference paper

    columns_permutation = None
    squared_norms = None
    max_norm = None
    norms_error_estimate = None
    if (
        column_pivoting
    ):  # Initialize variables for fast pivoting, without re-evaluation of the norm at each step.
        squared_norms = weighted_norm(Q, M=M) ** 2
        max_norm = np.sqrt(np.max(squared_norms))
        columns_permutation = np.arange(num_vecs)

    for k in range(num_vecs):
        if column_pivoting:
            Q, R, squared_norms, columns_permutation = _column_pivot(
                Q, R, k, squared_norms, columns_permutation
            )
            norms_error_estimate = squared_norms * _roundoff
        if (
            k != 0
        ):  # Reorthogonalization of the column k+1 of A with respect to the previous orthonormal k vectors.
            alpha = weighted_dot_product(
                Q[:, :k], Q[:, k], M=M
            )  # alpha = Q[:,:k].T@M@Q[:,k]
            R[:k, k] += alpha
            Q[:, k] -= np.dot(Q[:, :k], alpha)

        # Numerical rank detection, performed only when column_pivoting == True
        norm_at_iter_k = weighted_norm(Q[:, k], M=M)
        if column_pivoting:
            if norm_at_iter_k < rtol * max_norm:
                effective_rank = k
                if verbose:
                    warn(
                        "Numerical rank of A has been reached with a relative tolerance rtol = {:.2e}. "
                        "Effective rank = {}. Stopping Orthogonalization procedure.".format(
                            rtol, effective_rank
                        )
                    )
                break
                # Normalization of the column k + 1
        R[k, k] = norm_at_iter_k
        Q[:, k] = Q[:, k] / R[k, k]
        # Orthogonalization of the remaining columns with respect to Q[:,k], i.e. the k+1 column of Q.
        if k < num_vecs - 1:
            R[k, k + 1 :] = weighted_dot_product(Q[:, k + 1 :], Q[:, k], M=M)
            Q[:, k + 1 :] -= np.outer(Q[:, k], R[k, k + 1 :])
            # Try fast update of the squared norms, recompute if numerical criteria are not attained.
            if column_pivoting:
                squared_norms[k + 1 :] -= (
                    R[k, k + 1 :] ** 2
                )  # Update norms using Phythagorean Theorem
                update_error_mask = (
                    _tau * squared_norms[k + 1 :] < norms_error_estimate[k + 1 :]
                )  # Check if the error estimate is too large
                if any(update_error_mask):
                    squared_norms[k + 1 :][update_error_mask] = weighted_norm(
                        Q[:, k + 1 :][:, update_error_mask], M=M
                    )  # Recompute the norms if necessary.
    if column_pivoting:
        return (
            Q[:, :effective_rank],
            R[:effective_rank],
            columns_permutation[:effective_rank],
        )
    else:
        return Q[:, :effective_rank], R[:effective_rank]


def eigh_rank_reveal(
    values: np.ndarray,
    vectors: np.ndarray,
    rank: int,  # Desired rank
    rcond: Optional[float] = None,  # Threshold for the singular values
    verbose: bool = True,
):
    if rcond is None:
        rcond = 10.0 * values.shape[0] * np.finfo(values.dtype).eps
    top_vals = topk(values, rank)
    vectors = vectors[:, top_vals.indices]
    values = top_vals.values

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

        if verbose:
            logger.warning(
                f"Warning: Discarted {rank - vectors.shape[1]} dimensions of the {rank} requested due to numerical instability. Consider decreasing the rank. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        # Compute stable sqrt
        rsqrt_vals = (np.sqrt(values)) ** -1
    return vectors, values, rsqrt_vals


def cov(X: np.ndarray, Y: Optional[np.ndarray] = None):
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
