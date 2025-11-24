"""Loss functions for Koopman operator learning with JAX.

This module provides differentiable loss functions commonly used in
Koopman operator learning, including:

- VAMP score variants
- Spectral contrastive loss
- Dynamic autoencoder losses
- Orthonormality regularization terms

All functions are JAX-compatible and support automatic differentiation.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from kooplearn.jax.nn._linalg import sqrtmh
from kooplearn.jax.nn._stats import cov_norm_squared_unbiased, covariance


def vamp_loss(
    x: ArrayLike,
    y: ArrayLike,
    schatten_norm: int = 2,
    center_covariances: bool = True,
) -> jax.Array:
    r"""Variational Approach for learning Markov Processes (VAMP) score.

    Computes the negative VAMP-p score as introduced by :cite:t:`vamp_loss-Wu2019`.
    The VAMP score measures the quality of a feature transformation by quantifying
    how well it captures slow processes in the dynamics.

    .. math::

        \mathcal{L}(x, y) = -\sum_{i} \sigma_{i}(A)^{p}

    where

    .. math::

        A = (x^{\top}x)^{\dagger/2} x^{\top}y (y^{\top}y)^{\dagger/2}

    and :math:`\sigma_i(A)` are the singular values of A.

    Parameters
    ----------
    x : ArrayLike
        Input features of shape `(N, D_x)`, where N is the number of samples
        and D_x is the input feature dimension.
    y : ArrayLike
        Output features of shape `(N, D_y)`, where N is the number of samples
        and D_y is the output feature dimension.
    schatten_norm : int, optional
        Order p of the Schatten norm. Computes the VAMP-p score.
        Currently supports p=1 (nuclear norm) and p=2 (Frobenius norm).
        Default is 2.
    center_covariances : bool, optional
        If True, use centered covariances (subtract means). If False, use
        uncentered covariances. Default is True.

    Returns
    -------
    jax.Array
        Scalar loss value (negative VAMP-p score).

    Raises
    ------
    NotImplementedError
        If `schatten_norm` is not 1 or 2.

    Notes
    -----
    For p=2, a numerically stable least-squares formulation is used instead
    of direct pseudoinverse computation.

    References
    ----------
    .. cite:t:`vamp_loss-Wu2019`
    """
    cov_x, cov_y, cov_xy = (
        covariance(x, center=center_covariances),
        covariance(y, center=center_covariances),
        covariance(x, y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_x = jnp.linalg.lstsq(cov_x, cov_xy)[0]
        M_y = jnp.linalg.lstsq(cov_y, cov_xy.T)[0]
        return -jnp.trace(M_x @ M_y)
    elif schatten_norm == 1:
        sqrt_cov_x = sqrtmh(cov_x)
        sqrt_cov_y = sqrtmh(cov_y)
        M = jnp.linalg.multi_dot(
            [
                jnp.linalg.pinv(sqrt_cov_x, hermitian=True),
                cov_xy,
                jnp.linalg.pinv(sqrt_cov_y, hermitian=True),
            ]
        )
        return -jnp.linalg.norm(M, "nuc")
    else:
        raise NotImplementedError(
            f"Schatten norm {schatten_norm} not implemented. "
            "Supported values are 1 and 2."
        )


def spectral_contrastive_loss(x: ArrayLike, y: ArrayLike) -> jax.Array:
    r"""Spectral contrastive loss for self-supervised learning.

    Originally introduced by :cite:t:`spectral_contrastive_loss-haochen2021provable`
    and adapted for evolution operators in
    :cite:t:`spectral_contrastive_loss-turri2025self` and
    :cite:t:`spectral_contrastive_loss-jeong2025efficient`.

    The loss encourages alignment of paired samples (x_i, y_i) while
    discouraging alignment of unpaired samples:

    .. math::

        \mathcal{L}(x, y) = \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle^2
                            - \frac{2}{N}\sum_{i=1}^N\langle x_{i}, y_{i} \rangle

    Parameters
    ----------
    x : ArrayLike
        Input features of shape `(N, D)`, where N is the number of samples
        and D is the feature dimension.
    y : ArrayLike
        Output features of shape `(N, D)`. Must have the same shape as `x`.

    Returns
    -------
    jax.Array
        Scalar loss value.

    Raises
    ------
    ValueError
        If x and y do not have the same shape or if x is not 2-dimensional.

    References
    ----------
    .. cite:t:`spectral_contrastive_loss-haochen2021provable`
    .. cite:t:`spectral_contrastive_loss-turri2025self`
    .. cite:t:`spectral_contrastive_loss-jeong2025efficient`
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )
    if x.ndim != 2:
        raise ValueError(f"x must be 2-dimensional, got {x.ndim} dimensions")

    npts, dim = x.shape
    diag = 2 * jnp.mean(x * y) * dim
    square_term = (x @ y.T) ** 2
    off_diag = (
        jnp.mean(jnp.triu(square_term, k=1) + jnp.tril(square_term, k=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - diag


def autoencoder_loss(
    x: ArrayLike,
    y: ArrayLike,
    x_rec: ArrayLike,
    y_enc: ArrayLike,
    x_evo: ArrayLike,
    y_pred: ArrayLike,
    alpha_rec: float = 1.0,
    alpha_lin: float = 1.0,
    alpha_pred: float = 1.0,
) -> jax.Array:
    r"""Single-step Dynamic Autoencoder (DAE) loss.

    Introduced by :cite:t:`autoencoder_loss-Lusch2018`. This loss combines three
    objectives to train dynamic autoencoders for learning Koopman operators:

    1. **Reconstruction loss**: Measures how well the autoencoder reconstructs inputs
    2. **Linearity loss**: Enforces linear evolution in the latent space
    3. **Prediction loss**: Measures prediction quality in the original space

    The total loss is:

    .. math::

        \mathcal{L} = \alpha_\mathrm{rec} \|x - \phi^{-1}(\phi(x))\|^2
                    + \alpha_\mathrm{lin} \|\phi(y) - K\phi(x)\|^2
                    + \alpha_\mathrm{pred} \|y - \phi^{-1}(K\phi(x))\|^2

    where :math:`\phi` is the encoder, :math:`\phi^{-1}` is the decoder,
    and :math:`K` is the Koopman operator in latent space.

    Parameters
    ----------
    x : ArrayLike
        Input features of shape `(N, D)`, where N is the number of samples
        and D is the input dimension.
    y : ArrayLike
        Target output features of shape `(N, D)`.
    x_rec : ArrayLike
        Reconstructed input :math:`\phi^{-1}(\phi(x))` of shape `(N, D)`.
    y_enc : ArrayLike
        Encoded target :math:`\phi(y)` of shape `(N, d)`, where d is the
        latent dimension.
    x_evo : ArrayLike
        Evolved latent representation :math:`K\phi(x)` of shape `(N, d)`.
    y_pred : ArrayLike
        Predicted decoded output :math:`\phi^{-1}(K\phi(x))` of shape `(N, D)`.
    alpha_rec : float, optional
        Weight for the reconstruction term. Default is 1.0.
    alpha_lin : float, optional
        Weight for the linearity term. Default is 1.0.
    alpha_pred : float, optional
        Weight for the prediction term. Default is 1.0.

    Returns
    -------
    jax.Array
        Scalar total loss value.

    References
    ----------
    .. cite:t:`autoencoder_loss-Lusch2018`
    """

    def mse(true, pred):
        return jnp.mean((true - pred) ** 2)

    rec_loss = mse(x, x_rec)
    lin_loss = mse(y_enc, x_evo)
    pred_loss = mse(y, y_pred)
    return alpha_rec * rec_loss + alpha_lin * lin_loss + alpha_pred * pred_loss


def orthonormal_fro_reg(x: ArrayLike, key: jax.random.PRNGKey) -> jax.Array:
    r"""Orthonormality regularization using Frobenius norm.

    Encourages the features to have an identity covariance matrix and zero mean.
    This regularization promotes orthonormal representations in the latent space.

    The regularization term is:

    .. math::

        \frac{1}{D} \left( \|\mathbf{C}_{X} - I\|_F^2 + 2\|\mathbb{E}[x]\|^2 \right)

    where :math:`\mathbf{C}_X` is the covariance matrix of x and D is the
    feature dimension.

    Parameters
    ----------
    x : ArrayLike
        Input features of shape `(N, D)`, where N is the number of samples
        and D is the feature dimension.
    key : jax.random.PRNGKey
        JAX random key used for unbiased covariance estimation via permutation.

    Returns
    -------
    jax.Array
        Scalar regularization value.

    Notes
    -----
    The covariance norm is computed using an unbiased estimator that requires
    random permutations, hence the need for a PRNG key.
    """
    x = jnp.asarray(x)
    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    Cx_fro_2 = cov_norm_squared_unbiased(x_centered, key=key)
    tr_Cx = jnp.einsum("ij,ij->", x_centered, x_centered) / x.shape[0]
    centering_loss = (x_mean**2).sum()
    D = x.shape[-1]
    reg = Cx_fro_2 - 2 * tr_Cx + D + 2 * centering_loss
    return reg / D


def orthonormal_logfro_reg(x: ArrayLike) -> jax.Array:
    r"""Orthonormality regularization using log-Frobenius norm.

    An alternative to `orthonormal_fro_reg` that uses a logarithmic penalty
    on the eigenvalues of the covariance matrix. This can provide better
    conditioning and avoid issues with very small or large eigenvalues.

    The regularization term is:

    .. math::

        \frac{1}{D}\text{Tr}(C_X^{2} - C_X - \ln(C_X)) + 2\|\mathbb{E}[x]\|^2

    where :math:`C_X` is the covariance matrix of x.

    Parameters
    ----------
    x : ArrayLike
        Input features of shape `(N, D)`, where N is the number of samples
        and D is the feature dimension.

    Returns
    -------
    jax.Array
        Scalar regularization value.

    Notes
    -----
    Eigenvalues below machine epsilon are clamped to avoid numerical issues
    with the logarithm.
    """
    x = jnp.asarray(x)
    cov = covariance(x)  # shape: (D, D)
    eps = jnp.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = jnp.linalg.eigvalsh(cov)
    vals_x = jnp.where(vals_x > eps, vals_x, eps)
    orth_loss = jnp.mean(-jnp.log(vals_x) + vals_x * (vals_x - 1.0))
    centering_loss = (x.mean(0, keepdims=True) ** 2).sum()
    reg = orth_loss + 2 * centering_loss
    return reg
