from kooplearn._src.check_deps import check_torch_deps
check_torch_deps()
import torch  # noqa: E402

def sqrtmh(A):
    #Credits to https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices"""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def VAMP_score(cov_X, cov_Y, cov_XY, schatten_norm: int = 2):
    if schatten_norm == 2:
        M = torch.linalg.multi_dot([torch.linalg.pinv(cov_X, hermitian=True), cov_XY, torch.linalg.pinv(cov_Y, hermitian=True), cov_XY.T])
        return torch.trace(M)
    elif schatten_norm == 1:
        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)
        M = torch.linalg.multi_dot([torch.linalg.pinv(sqrt_cov_X, hermitian=True), cov_XY, torch.linalg.pinv(sqrt_cov_Y, hermitian=True)])
        return torch.linalg.matrix_norm(M, 'nuc')
    else:
        raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")
