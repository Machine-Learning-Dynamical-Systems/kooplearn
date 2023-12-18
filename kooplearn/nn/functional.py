from kooplearn._src.check_deps import check_torch_deps

check_torch_deps()
import torch  # noqa: E402


def sqrtmh(A):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def vamp_score(cov_X, cov_Y, cov_XY, schatten_norm: int = 2):
    """Variational Approach for learning Markov Processes (VAMP) score by :footcite:t:`Wu2019`.

    Args:
        cov_X (torch.Tensor): Covariance of the initial time steps.
        cov_Y (torch.Tensor): Covariance of the evolved time steps.
        cov_XY (torch.Tensor): Cross-covariance of the initial and evolved time steps.
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.

    Raises:
        NotImplementedError: If ``schatten_norm`` is not 1 or 2.

    """
    if schatten_norm == 2:
        M = torch.linalg.multi_dot(
            [
                torch.linalg.pinv(cov_X, hermitian=True),
                cov_XY,
                torch.linalg.pinv(cov_Y, hermitian=True),
                cov_XY.T,
            ]
        )
        return torch.trace(M)
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


def relaxed_projection_score(cov_x, cov_y, cov_xy):
    return (torch.linalg.matrix_norm(cov_xy, ord="fro") ** 2) / (
        (
            torch.linalg.matrix_norm(cov_x, ord=2)
            * torch.linalg.matrix_norm(cov_y, ord=2)
        )
    )


def log_fro_metric_deformation_loss(cov):
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    # Metric regularization based on Von Neumann entropy
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss
