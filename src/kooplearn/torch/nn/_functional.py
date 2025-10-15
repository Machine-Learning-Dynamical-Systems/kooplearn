"""Functional interface."""

import torch
from torch import Tensor

from ._linalg import sqrtmh
from ._stats import cov_norm_squared_unbiased, covariance


# Losses_______________________________________________________________________

def vamp_loss(
    x: Tensor, y: Tensor, schatten_norm: int = 2, center_covariances: bool = True
) -> Tensor:
    """See :class:`kooplearn.torch.nn.VampLoss` for details."""
    cov_x, cov_y, cov_xy = (
        covariance(x, center=center_covariances),
        covariance(y, center=center_covariances),
        covariance(x, y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_x = torch.linalg.lstsq(cov_x, cov_xy).solution
        M_y = torch.linalg.lstsq(cov_y, cov_xy.T).solution
        return -torch.trace(M_x @ M_y)
    elif schatten_norm == 1:
        sqrt_cov_x = sqrtmh(cov_x)
        sqrt_cov_y = sqrtmh(cov_y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_x, hermitian=True),
                cov_xy,
                torch.linalg.pinv(sqrt_cov_y, hermitian=True),
            ]
        )
        return -torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def l2_contrastive_loss(x: Tensor, y: Tensor) -> Tensor:
    """See :class:`kooplearn.torch.nn.L2ContrastiveLoss` for details."""
    assert x.shape == y.shape
    assert x.ndim == 2

    npts, dim = x.shape
    diag = 2 * torch.mean(x * y) * dim
    square_term = torch.matmul(x, y.T) ** 2
    off_diag = (
        torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - diag


def kl_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    """See :class:`kooplearn.torch.nn.KLContrastiveLoss` for details."""
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    log_term = torch.mean(torch.log(X * Y)) * dim
    linear_term = torch.matmul(X, Y.T)
    off_diag = (
        torch.mean(torch.triu(linear_term, diagonal=1) + torch.tril(linear_term, diagonal=-1))
        * npts
        / (npts - 1)
    )
    return off_diag - log_term


# Regularizers_________________________________________________________________

def orthonormal_fro_reg(x: Tensor) -> Tensor:
    r"""Orthonormality regularization with Frobenious norm of covariance of `x`.

    Given a batch of realizations of `x`, the orthonormality regularization term penalizes:

    1. Orthogonality: Linear dependencies among dimensions,
    2. Normality: Deviations of each dimension’s variance from 1,
    3. Centering: Deviations of each dimension’s mean from 0.

    .. math::

        \frac{1}{D} \| \mathbf{C}_{X} - I \|_F^2 +  2 \| \mathbb{E}_{X} x \|^2 = \frac{1}{D} (\text{tr}(\mathbf{C}^2_{X}) - 2 \text{tr}(\mathbf{C}_{X}) + D + 2 \| \mathbb{E}_{X} x \|^2)

    Args:
        x (Tensor): Input features.

    Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    # As ||Cx||_F^2 = E_(x,x')~p(x) [((x - E_p(x) x)^T (x' - E_p(x) x'))^2] = tr(Cx^2), involves the product of
    # covariances, unbiased estimation of this term requires the use of U-statistics
    Cx_fro_2 = cov_norm_squared_unbiased(x_centered)
    # tr(Cx) = E_p(x) [(x - E_p(x))^T (x - E_p(x))] ≈ 1/N Σ_n (x_n - E_p(x))^T (x_n - E_p(x))
    tr_Cx = torch.einsum("ij,ij->", x_centered, x_centered) / x.shape[0]
    centering_loss = (x_mean**2).sum()  # ||E_p(x) x||^2
    D = x.shape[-1]  # ||I||_F^2 = D
    reg = Cx_fro_2 - 2 * tr_Cx + D + 2 * centering_loss
    return reg / D


def orthonormal_logfro_reg(x: Tensor) -> Tensor:
    r"""Orthonormality regularization with log-Frobenious norm of covariance of x by :footcite:t:`Kostic2023DPNets`.

    .. math::

        \frac{1}{D}\text{Tr}(C_X^{2} - C_X -\ln(C_X)).

    Args:
        x (Tensor): Input features.

    Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    cov = covariance(x)  # shape: (D, D)
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    orth_loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    # TODO: Centering like this?
    centering_loss = (x.mean(0, keepdim=True) ** 2).sum()  # ||E_p(x) x||^2
    reg = orth_loss + 2 * centering_loss
    return reg
