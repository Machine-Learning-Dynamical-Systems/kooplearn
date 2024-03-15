from kooplearn._src.check_deps import check_torch_deps

check_torch_deps()
from typing import Optional  # noqa: E402

import torch  # noqa: E402


def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def covariance(X: torch.Tensor, Y: Optional[torch.Tensor] = None, center: bool = True):
    """Covariance matrix

    Args:
        X (torch.Tensor): Input covariates of shape ``(samples, features)``.
        Y (Optional[torch.Tensor], optional): Output covariates of shape ``(samples, features)`` Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.

    Returns:
        torch.Tensor: Covariance matrix of shape ``(features, features)``. If ``Y is not None`` computes the cross-covariance between X and Y.
    """
    assert X.ndim == 2
    cov_norm = torch.rsqrt(torch.tensor(X.shape[0]))
    if Y is None:
        _X = cov_norm * X
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _X)
    else:
        assert Y.ndim == 2
        _X = cov_norm * X
        _Y = cov_norm * Y
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
            _Y = _Y - _Y.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _Y)


def vamp_score(X, Y, schatten_norm: int = 2, center_covariances: bool = True):
    """Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    Raises:
        NotImplementedError: If ``schatten_norm`` is not 1 or 2.

    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    if schatten_norm == 2:
        # Using least squares in place of pinv for numerical stability
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        return torch.trace(M_X @ M_Y)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(sqrt_cov_X, hermitian=True),
                cov_XY,
                torch.linalg.pinv(sqrt_cov_Y, hermitian=True),
            ]
        )
        return torch.linalg.matrix_norm(M, "nuc")
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")


def deepprojection_score(
    X,
    Y,
    relaxed: bool = True,
    metric_deformation: float = 1.0,
    center_covariances: bool = True,
):
    """Deep Projection score by :footcite:t:`Kostic2023DPNets`.

    Args:
        X (torch.Tensor): Covariates for the initial time steps.
        Y (torch.Tensor): Covariates for the evolved time steps.
        relaxed (bool, optional): Whether to use the relaxed (more numerically stable) or the full deep-projection loss. Defaults to True.
        metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

    """
    cov_X, cov_Y, cov_XY = (
        covariance(X, center=center_covariances),
        covariance(Y, center=center_covariances),
        covariance(X, Y, center=center_covariances),
    )
    R_X = log_fro_metric_deformation_loss(cov_X)
    R_Y = log_fro_metric_deformation_loss(cov_Y)
    if relaxed:
        S = (torch.linalg.matrix_norm(cov_XY, ord="fro") ** 2) / (
            (
                torch.linalg.matrix_norm(cov_X, ord=2)
                * torch.linalg.matrix_norm(cov_Y, ord=2)
            )
        )
    else:
        M_X = torch.linalg.lstsq(cov_X, cov_XY).solution
        M_Y = torch.linalg.lstsq(cov_Y, cov_XY.T).solution
        S = torch.trace(M_X @ M_Y)
    return S - 0.5 * metric_deformation * (R_X + R_Y)


def log_fro_metric_deformation_loss(cov: torch.tensor):
    """Logarithmic + Frobenious metric deformation loss as used in :footcite:t:`Kostic2023DPNets`, defined as :math:`{{\\rm Tr}}(C^{2} - C -\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix.

    Returns:
        torch.tensor: Loss function
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss
