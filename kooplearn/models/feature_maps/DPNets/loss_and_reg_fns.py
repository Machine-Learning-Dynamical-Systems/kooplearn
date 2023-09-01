import torch
import torch.linalg

def projection_loss(cov_x, cov_y, cov_xy):
    M = torch.linalg.multi_dot([torch.linalg.pinv(cov_x, hermitian=True), cov_xy, torch.linalg.pinv(cov_y, hermitian=True), cov_xy.T])
    return -1.0*torch.trace(M).mean()

def relaxed_projection_loss(cov_x, cov_y, cov_xy):
    return -1.0*((torch.linalg.matrix_norm(cov_xy, ord='fro')**2) / ((torch.linalg.matrix_norm(cov_x, ord=2) * torch.linalg.matrix_norm(cov_y, ord=2)))).mean()

def fro_reg(cov_x, cov_y):
    identity = torch.eye(cov_x.shape[0], device=cov_x.device)
    r1 = torch.linalg.matrix_norm(identity - cov_x, ord='fro')
    r2 = torch.linalg.matrix_norm(identity - cov_y, ord='fro')
    return 0.5*(r1 + r2).mean()

def von_neumann_reg(cov_x, cov_y):
    eps = torch.finfo(cov_x.dtype).eps*cov_x.shape[0]
    # Metric regularization based on Von Neumann entropy
    vals_x = torch.linalg.eigvalsh(cov_x)
    vals_x = torch.where(vals_x > eps, vals_x, 0)
    r_1 = torch.mean(1 - vals_x - torch.special.entr(vals_x))

    vals_y = torch.linalg.eigvalsh(cov_y)
    vals_y = torch.where(vals_y > eps, vals_y, 0)
    r_2 = torch.mean(1 - vals_y - torch.special.entr(vals_y))

    return 0.5*(r_1 + r_2).mean()

def log_fro_reg(cov_x, cov_y):
    eps = torch.finfo(cov_x.dtype).eps*cov_x.shape[0]
    # Metric regularization based on Von Neumann entropy
    vals_x = torch.linalg.eigvalsh(cov_x)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    r_1 = torch.mean(-torch.log(vals_x) + vals_x*(vals_x - 1.0))

    vals_y = torch.linalg.eigvalsh(cov_y)
    vals_y = torch.where(vals_y > eps, vals_y, eps)
    r_2 = torch.mean(-torch.log(vals_y) + vals_y*(vals_y - 1.0))

    return 0.5*(r_1 + r_2).mean()