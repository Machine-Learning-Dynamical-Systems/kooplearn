"""Statistics utilities for multi-variate random variables."""

from math import sqrt
from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def covariance(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    center: bool = True,
    norm: Optional[float] = None,
) -> ArrayLike:
    """Computes the covariance of X or cross-covariance between X and Y if Y is given.

    Args:
        X (ArrayLike): Input features of shape `(N, D)`.
        Y (ArrayLike, optional): Output features of shape `(N, D)`. Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.
        norm (float, optional): Normalization factor. Defaults to `sqrt(N)`.

    Returns:
        ArrayLike: Covariance matrix of shape `(D, D)`.
    """
    X = jnp.asarray(X)
    assert X.ndim == 2
    if norm is None:
        norm = sqrt(X.shape[0])
    else:
        assert norm > 0
        norm = sqrt(norm)

    X = X / norm
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    if Y is None:
        return X.T @ X
    else:
        Y = jnp.asarray(Y)
        assert Y.ndim == 2
        Y = Y / norm
        if center:
            Y = Y - Y.mean(axis=0, keepdims=True)
        return X.T @ Y


def cross_cov_norm_squared_unbiased(
    x: ArrayLike, y: ArrayLike, key: jax.random.PRNGKey, permutation=None
):
    r"""Compute the unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2` from a batch of samples, using U-statistics.

    Given the Covariance matrix :math:`\mathbf{C}_{xy} = \mathbb{E}_p(x,y) [x^{\top} y]`, this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from two independent sampling sets (an effective samples size of :math:`N^2`).

    .. math::

        \begin{align}
            \|\mathbf{C}_{xy}\|_F^2 &= \text{tr}(\mathbf{C}_{xy}^{\top} \mathbf{C}_{xy})
            = \sum_i \sum_j (\mathbb{E}_{x,y \sim p(x,y)} [x_i y_j]) (\mathbb{E}_{x',y' \sim p(x,y)} [x_j y_i']) \\
            &= \mathbb{E}_{(x,y),(x',y') \sim p(x,y)} [(x^{\top} y') (x'^{\top} y)] \\
            &\approx \frac{1}{N^2} \sum_n \sum_m [(x_{n}^{\top} y^{\prime}_m) (x^{\prime \top}_m y_n)]
        \end{align}

    .. note::
    The random variable is assumed to be centered.

    Args:
        x (ArrayLike): Centered realizations of a random variable `x` of shape (N, D_x).
        y (ArrayLike): Centered realizations of a random variable `y` of shape (N, D_y).
        key (jax.random.PRNGKey): JAX random key for permutation.
        permutation (ArrayLike, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        ArrayLike: Unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2` using U-statistics.
    """
    n_samples = x.shape[0]

    # Permute the rows independently to simulate independent sampling
    if permutation is None:
        perm = jax.random.permutation(key, n_samples)
    else:
        perm = permutation
    assert perm.shape == (
        n_samples,
    ), f"Invalid permutation {perm.shape}!=({n_samples},)"
    xp = x[perm]  # Independent sampling of x'
    yp = y[perm]  # Independent sampling of y'

    # Compute 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]
    val = jnp.einsum("nj,mj,mk,nk->", x, yp, xp, y)
    cov_fro_norm = val / (n_samples**2)
    return cov_fro_norm


def cov_norm_squared_unbiased(x: ArrayLike, key: jax.random.PRNGKey, permutation=None):
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
        x (ArrayLike): (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        key (jax.random.PRNGKey): JAX random key for permutation.
        permutation (ArrayLike, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        ArrayLike: Unbiased estimation of :math:`\|\mathbf{C}_x\|_F^2` using U-statistics.
    """
    return cross_cov_norm_squared_unbiased(x=x, y=x, key=key, permutation=permutation)
