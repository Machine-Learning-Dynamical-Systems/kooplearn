import torch

def VAMP_score(cov_X, cov_Y, cov_XY, schatten_norm: int = 2):
    if schatten_norm == 2:
        M = torch.linalg.multi_dot([torch.linalg.pinv(cov_X, hermitian=True), cov_XY, torch.linalg.pinv(cov_Y, hermitian=True), cov_XY.T])
        return torch.trace(M).mean()
    else:
        raise NotImplementedError('Only schatten_norm = 2 is supported at the moment.')