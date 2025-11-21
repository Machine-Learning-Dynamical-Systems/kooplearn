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


def cross_covariance(A, B, rowvar=True, bias=False, centered=True):
    """Cross covariance of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the covariance along the rows. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross covariance of A and B.
    """
    if rowvar is False:
        A = A.T
        B = B.T

    if centered:
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)

    C = A @ B.T

    if bias:
        return C / A.shape[1]
    else:
        return C / (A.shape[1] - 1)


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


def eym_score_Ustat(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric_deformation: float = 1.0,
    center: bool = True,
):
    if center:
        _X = X - X.mean(0, keepdim=True)
        _Y = Y - Y.mean(0, keepdim=True)
    else:
        _X = X
        _Y = Y

    joint_measure_score = ((_X * _X).sum(dim=-1)).mean() + (
        (_Y * _Y).sum(dim=-1)
    ).mean()
    product_measure_score = torch.square_(_X @ _Y.T)
    product_measure_score = product_measure_score.fill_diagonal_(
        0
    )  # Zeroing out the diagonal in place
    b = X.shape[0]
    product_measure_score = b * product_measure_score.mean() / (b - 1)
    score = joint_measure_score - product_measure_score
    return score


def eym_score_split(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric_deformation: float = 1.0,
):
    if X.shape[0] // 2 != 0:
        X = X[:-1]
        Y = Y[:-1]
    batch_size = X.shape[0]
    # Randomly splitting in half
    idxs = torch.randperm(batch_size)
    U1 = X[idxs[: batch_size // 2]]
    U2 = X[idxs[batch_size // 2 :]]

    V1 = Y[idxs[: batch_size // 2]]
    V2 = Y[idxs[batch_size // 2 :]]

    # centered covariance matrices
    cov_U1 = torch.cov(U1.T)
    cov_U2 = torch.cov(U2.T)
    cov_V1 = torch.cov(V1.T)
    cov_V2 = torch.cov(V2.T)

    cov_U1V1 = cross_covariance(U1.T, V1.T, centered=True)
    cov_U2V2 = cross_covariance(U2.T, V2.T, centered=True)

    score = -1 * (
        0.5 * (torch.trace(cov_U1 @ cov_V2) + torch.trace(cov_U2 @ cov_V1))
        - torch.trace(cov_U1V1)
        - torch.trace(cov_U2V2)
    )
    if score > 0:
        d = U1.shape[-1]
        U1_mean = U1.mean(axis=0, keepdims=True)
        U2_mean = U2.mean(axis=0, keepdims=True)
        V1_mean = V1.mean(axis=0, keepdims=True)
        V2_mean = V2.mean(axis=0, keepdims=True)

        # uncentered covariance matrices
        uc_cov_U1 = cov_U1 + U1_mean @ U1_mean.T
        uc_cov_U2 = cov_U2 + U2_mean @ U2_mean.T
        uc_cov_V1 = cov_V1 + V1_mean @ V1_mean.T
        uc_cov_V2 = cov_V2 + V2_mean @ V2_mean.T

        loss_on = (
            0.5
            * (
                torch.trace(uc_cov_U1 @ uc_cov_U2)
                - torch.trace(uc_cov_U1)
                - torch.trace(uc_cov_U2)
                + torch.trace(uc_cov_V1 @ uc_cov_V2)
                - torch.trace(uc_cov_V1)
                - torch.trace(uc_cov_V2)
            )
            + d
        )
        return score - metric_deformation * loss_on
    else:
        return score


def eym_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    mode: str = "split",
    metric_deformation: float = 1.0,
    center: bool = True,
):
    available_modes = ["split", "U_stat"]
    if mode not in available_modes:
        raise ValueError(f"Unknown mode {mode}. Available modes are {available_modes}")
    if mode == "split":
        return eym_score_split(X, Y, metric_deformation=metric_deformation)
    else:
        return eym_score_Ustat(
            X, Y, metric_deformation=metric_deformation, center=center
        )


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
