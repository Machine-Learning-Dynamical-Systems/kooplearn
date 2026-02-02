"""Linear algebra utilities for the `kernel` algorithms."""

from warnings import warn

import numpy as np
from numpy import ndarray

from kooplearn._utils import stable_topk


def add_diagonal_(M: ndarray, alpha: float):
    """Add alpha to the diagonal of M inplace.

    Parameters
    ----------
    M : ndarray
        The matrix to modify inplace.
    alpha : float
        The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def spd_neg_pow(
    M: np.ndarray,
    exponent: float = -1.0,
    cutoff: float | None = None,
    strategy: str = "trunc",
) -> np.ndarray:
    """Computes the negative power of a symmetric positive definite (SPD) matrix.

    This function computes :math:`M^{\alpha}` where :math:`\alpha` is the ``exponent``,
    using a truncated eigenvalue decomposition.

    Parameters
    ----------
    M : np.ndarray
        The symmetric positive definite matrix.
    exponent : float, optional
        The exponent to which the matrix is raised. Default is -1.0.
    cutoff : float or None, optional
        The cutoff for the eigenvalues. Eigenvalues below this value are treated
        as zero. If None, a default value is computed based on the machine
        epsilon. Default is None.
    strategy : str, optional
        The strategy to handle small eigenvalues. Can be 'trunc' (truncation) or
        'tikhonov' (Tikhonov regularization). Default is "trunc".

    Returns
    -------
    np.ndarray
        The matrix :math:`M` raised to the power of ``exponent``.
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


def covariance(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute the covariance matrix of two arrays of observations.

    This function computes the covariance matrix between two arrays of observations `X`
    and `Y`. The observations are assumed to be stored in the rows of the arrays.
    The normalization is performed by dividing by the number of observations, `n`,
    i.e. :math:`C_{XY} = n^{-1} X^T Y`.

    Parameters
    ----------
    X : np.ndarray
        An array of shape `(n, d_x)` where `n` is the number of observations and
        `d_x` is the number of features.
    Y : np.ndarray or None, optional
        An array of shape `(n, d_y)` where `n` is the number of observations and
        `d_y` is the number of features. If `None`, the covariance of `X` with
        itself is computed. Default is None.

    Returns
    -------
    np.ndarray
        The covariance matrix of shape `(d_x, d_y)`. If `Y` is `None`, the shape is
        `(d_x, d_x)`.

    Raises
    ------
    ValueError
        If `X` or `Y` have more than 2 dimensions, or if `X` and `Y` do
        not have the same number of observations.
    """
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
                "Shape mismatch: the covariance between two arrays can be computed only if they " /
                f"have the same initial dimension. Got {X.shape[0]} and {Y.shape[0]}."
            )
        Y = np.atleast_2d(Y)
        if Y.ndim > 2:
            raise ValueError(f"Input array has more than 2 dimensions ({Y.ndim}).")
        Y = Y * rnorm
        c = X.T @ Y
    return c


def weighted_norm(A: ndarray, M: ndarray | None = None) -> float | np.ndarray:
    r"""Weighted norm of the columns of A.

    Parameters
    ----------
    A : ndarray
        1D or 2D array. If 2D, the columns are treated as vectors.
    M : ndarray or None, optional
        Weighting matrix. The norm of the vector :math:`a` is given by
        :math:`\langle a, Ma \rangle`. Default is None, corresponding to the
        Identity matrix. Warning: no checks are performed on M being a PSD
        operator.

    Returns
    -------
    ndarray or float
        If ``A.ndim == 2`` returns 1D array of floats corresponding to the norms
        of the columns of A. Else return a float.
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
    rank: int,
    rcond: float | None = None,
    ignore_warnings: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the top `rank` eigenvalues and eigenvectors and filter them.

    This function selects the top `rank` eigenvalues and eigenvectors based on the
    magnitude of the eigenvalues. It then filters out eigenvalues below a certain
    threshold `rcond`.

    Parameters
    ----------
    values : np.ndarray
        An array of eigenvalues.
    vectors : np.ndarray
        An array of eigenvectors, with each column representing an eigenvector.
    rank : int
        The desired rank (number of eigenvalues/eigenvectors to select).
    rcond : float or None, optional
        The threshold for the eigenvalues. Eigenvalues with magnitude below this
        value are discarded. If `None`, a default value is computed based on
        machine epsilon. Default is None.
    ignore_warnings : bool, optional
        If `False`, a warning is issued if any of the selected `rank`
        eigenvalues are discarded. Default is True.

    Returns
    -------
    vectors : np.ndarray
        The filtered eigenvectors (as columns of a 2D array).
    values : np.ndarray
        The filtered eigenvalues.
    rsqrt_vals : np.ndarray
        The reciprocal of the square root of the filtered eigenvalues.
    """
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
        )  # In the case of multiple occurrences of the maximum values, the indices corresponding to
        #the first occurrence are returned.
        _first_discarded_val = np.max(np.abs(values[first_invalid:]))
        values = values[_ftest]
        vectors = vectors[:, _ftest]

        if not ignore_warnings:
            warn(
                f"Warning: Discarded {rank - vectors.shape[1]} dimensions of the {rank} " /
                "requested due to numerical instability. Consider decreasing the rank. The " /
                "largest discarded value is: {_first_discarded_val:.3e}."
            )
        # Compute stable sqrt
        rsqrt_vals = (np.sqrt(values)) ** -1
    return vectors, values, rsqrt_vals
