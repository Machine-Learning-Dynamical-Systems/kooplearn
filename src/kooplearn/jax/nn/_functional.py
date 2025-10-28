"""Functional interface."""

import jax
import jax.numpy as jnp

from kooplearn.jax.nn._linalg import sqrtmh
from kooplearn.jax.nn._stats import cov_norm_squared_unbiased, covariance


def vamp_loss(x, y, schatten_norm: int = 2, center_covariances: bool = True):
    r"""Variational Approach for learning Markov Processes (VAMP) score by :cite:t:`Wu2019`.

    .. math::

        \mathcal{L}(x, y) = -\sum_{i} \sigma_{i}(A)^{p} \qquad \text{where}~A = \big(x^{\top}x\big)^{\dagger/2}x^{\top}y\big(y^{\top}y\big)^{\dagger/2}.

    Args:
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.
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
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def spectral_contrastive_loss(x, y):
    r"""Spectral contrastive loss based originally introduced by :cite:t:`haochen2021provable`, and adopted for evolution operators in :cite:t:`turri2025self, jeong2025efficient`

    .. math::

        \mathcal{L}(x, y) = \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle^2 - \frac{2}{N}\sum_{i=1}\langle x_{i}, y_{i} \rangle.

    Args:
        x (ArrayLike): Input features, shape `(N, D)`.
        y (ArrayLike): Output features, shape `(N, D)`.

    Returns:
        ArrayLike: Spectral contrastive loss.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    assert x.shape == y.shape
    assert x.ndim == 2

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
    x,
    y,
    x_rec,
    y_enc,
    x_evo,
    y_pred,
    alpha_rec: float = 1.0,
    alpha_lin: float = 1.0,
    alpha_pred: float = 1.0,
):
    r"""Single-step Dynamic Autoencoder (DAE) loss introduced by :cite:t:`Lusch2018`.

    This loss combines three objectives to train dynamic autoencoders:

    1. **Reconstruction loss**
    2. **Linearity loss**
    3. **Prediction loss**

    The total loss is a weighted sum:

    .. math::
        \mathcal{L} =
        \alpha_\mathrm{rec} \, \|x - \phi^{-1}(\phi(x)) \|^2 +
        \alpha_\mathrm{lin} \, \|\phi(y) - K\phi(x) \|^2 +
        \alpha_\mathrm{pred} \, \|y - \phi^{-1}(K\phi(x))\|^2

    Args:
        x (ArrayLike): Input features, shape `(N, D)`.
        y (ArrayLike): Output (target) features, shape `(N, D)`.
        x_rec (ArrayLike): Reconstructed version of the input `x`.
        y_enc (ArrayLike): Encoded latent representation of the target `y`.
        x_evo (ArrayLike): Evolved latent representation.
        y_pred (ArrayLike): Predicted decoded output.
        alpha_rec (float, optional): Weight for the reconstruction term. Defaults to 1.0.
        alpha_lin (float, optional): Weight for the linearity term. Defaults to 1.0.
        alpha_pred (float, optional): Weight for the prediction term. Defaults to 1.0.

    Returns:
        ArrayLike: Total dynamic autoencoder loss.
    """

    def mse(true, pred):
        return jnp.mean((true - pred) ** 2)

    rec_loss = mse(x, x_rec)
    lin_loss = mse(y_enc, x_evo)
    pred_loss = mse(y, y_pred)
    return alpha_rec * rec_loss + alpha_lin * lin_loss + alpha_pred * pred_loss


def orthonormal_fro_reg(x, key: jax.random.PRNGKey):
    r"""Orthonormality regularization with Frobenious norm of covariance of `x`.

    .. math::

        \frac{1}{D} \| \mathbf{C}_{X} - I \|_F^2 +  2 \| \mathbb{E}_{X} x \|^2

    Args:
        x (ArrayLike): Input features, shape `(N, D)`.
        key (jax.random.PRNGKey): JAX random key for permutation.

    Returns:
        ArrayLike: Orthonormality regularization value.
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


def orthonormal_logfro_reg(x):
    r"""Orthonormality regularization with log-Frobenious norm of covariance of x.

    .. math::

        \frac{1}{D}\text{Tr}(C_X^{2} - C_X -\ln(C_X)).

    Args:
        x (ArrayLike): Input features, shape `(N, D)`.

    Returns:
        ArrayLike: Orthonormality regularization value.
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
