"""Statistics utilities for multi-variate random variables."""

from math import sqrt

import torch
from torch import Tensor

from ._linalg import filter_reduced_rank_svals, sqrtmh


def covariance(
    X: Tensor,
    Y: Tensor | None = None,
    center: bool = True,
    norm: float | None = None,
) -> Tensor:
    """Computes the covariance of X or cross-covariance between X and Y if Y is given.

    Args:
        X (Tensor): Input features.
        Y (Tensor | None, optional): Output features. Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.
        norm (float | None, optional): Normalization factor. Defaults to None.

    Shape:
        ``X``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        ``Y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

        Output: :math:`(D, D)`, where :math:`D` is the number of features.
    """
    assert X.ndim == 2
    if norm is None:
        norm = sqrt(X.shape[0])
    else:
        assert norm > 0
        norm = sqrt(norm)
    if Y is None:
        X = X / norm
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        return torch.mm(X.T, X)
    else:
        assert Y.ndim == 2
        X = X / norm
        Y = Y / norm
        if center:
            X = X - X.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)
        return torch.mm(X.T, Y)


def cross_cov_norm_squared_unbiased(x: Tensor, y: Tensor, permutation=None):
    r"""Compute the unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2` from a batch of samples, using U-statistics.

    Given the Covariance matrix :math:`\mathbf{C}_{xy} = \mathbb{E}_p(x,y) [x^{\top} y]`, this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from two independent sampling sets (an effective samples size of :math:`N^2`).

    .. math::

        \begin{align}
            \|\mathbf{C}_{xy}\|_F^2 &= \text{tr}(\mathbf{C}_{xy}^{\top} \mathbf{C}_{xy})
            = \sum_i \sum_j (\mathbb{E}_{x,y \sim p(x,y)} [x_i y_j]) (\mathbb{E}_{x',y' \sim p(x,y)} [x_j y_i']) \\
            &= \mathbb{E}_{(x,y),(x',y') \sim p(x,y)} [(x^{\top} y') (x'^{T} y)] \\
            &\approx \frac{1}{N^2} \sum_n \sum_m [(x_{n}^{\top} y^{\prime}_m) (x^{\prime \top}_m y_n)]
        \end{align}

    .. note::
    The random variable is assumed to be centered.

    Args:
        x (Tensor): Centered realizations of a random variable `x` of shape (N, D_x).
        y (Tensor): Centered realizations of a random variable `y` of shape (N, D_y).
        permutation (Tensor, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        Tensor: Unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2` using U-statistics.
    """
    n_samples = x.shape[0]

    # Permute the rows independently to simulate independent sampling
    perm = permutation if permutation is not None else torch.randperm(n_samples)
    assert perm.shape == (n_samples,), f"Invalid permutation {perm.shape}!=({n_samples},)"
    xp = x[perm]  # Independent sampling of x'
    yp = y[perm]  # Independent sampling of y'

    # Compute 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]
    val = torch.einsum("nj,mj,mk,nk->", x, yp, xp, y)
    cov_fro_norm = val / (n_samples**2)
    return cov_fro_norm


def cov_norm_squared_unbiased(x: Tensor, permutation=None):
    r"""Compute the unbiased estimation of :math:`\|\mathbf{C}_x\|_F^2` from a batch of samples.

    Given the Covariance matrix :math:`\mathbf{C}_x = \mathbb{E}_p(x) [x^{\top} x]`, this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from a single sampling set.

    .. math::

        \begin{align}
            \|\mathbf{C}_x\|_F^2 &= \text{tr}(\mathbf{C}_x^{\top} \mathbf{C}_x) = \sum_i \sum_j (\mathbb{E}_{x} [x_i x_j]) (\mathbb{E}_{x'} [x'_j x'_i]) \\
            &= \mathbb{E}_{x,x' \sim p(x)} [(x^{\top} x')^2] \\
            &\approx \frac{1}{N^2} \sum_n \sum_m [(x_n^{\top} x'_m)^2]
        \end{align}
    

    .. note::

        The random variable is assumed to be centered.

    Args:
        x (Tensor): (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        permutation (Tensor, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        Tensor: Unbiased estimation of :math:`\|\mathbf{C}_x\|_F^2` using U-statistics.
    """
    return cross_cov_norm_squared_unbiased(x=x, y=x, permutation=permutation)
