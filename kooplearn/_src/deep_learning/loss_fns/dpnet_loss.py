import torch


def dpnets_loss(x_encoded, y_encoded, rank=None, p_loss_coef=1.0, s_loss_coef=1.0, reg_1_coef=1.0, reg_2_coef=1.0):
    n = x_encoded.shape[-2]
    cov_x = x_encoded @ x_encoded.T / n
    cov_y = y_encoded @ y_encoded.T / n
    cov_xy = x_encoded @ y_encoded.T / n
    p_score = 0
    s_score = 0
    r1_score = 0
    r2_score = 0
    if p_loss_coef != 0:
        p_score = projection_score(cov_x, cov_y, cov_xy)
    if s_loss_coef != 0:
        s_score = spectral_score(cov_x, cov_y, cov_xy)
    if reg_1_coef != 0:
        r1_score = regularization_1(cov_x, cov_y)
    if reg_2_coef != 0:
        if rank is None:
            raise ValueError('rank must be specified for regularization 2')
        r2_score = regularization_2(cov_x, cov_y, rank)
    return -(p_loss_coef * p_score + s_loss_coef * s_score - reg_1_coef * r1_score - reg_2_coef * r2_score)


def projection_score(cov_x, cov_y, cov_xy):
    score = torch.linalg.lstsq(cov_x, cov_xy).solution  # == cov_x_inv @ cov_xy
    score = score @ torch.linalg.pinv(cov_y, hermitian=True)  # == cov_x_inv @ cov_xy @ cov_y_inv
    score = torch.linalg.matrix_norm(score, ord='fro')  # == ||cov_x_inv @ cov_xy @ cov_y_inv|| 2, HS
    return score.mean()


def spectral_score(cov_x, cov_y, cov_xy):
    score = torch.linalg.matrix_norm(cov_xy, ord='fro')  # == ||cov_xy|| 2, HS
    score = score / torch.linalg.matrix_norm(cov_x, ord=2)  # == ||cov_xy|| 2, HS / ||cov_x||
    score = score / torch.linalg.matrix_norm(cov_y, ord=2)  # == ||cov_xy|| 2, HS / (||cov_x|| * ||cov_y||)
    return score.mean()


def regularization_1(cov_x, cov_y):
    identity = torch.eye(cov_x.shape[0], device=cov_x.device)
    r1 = torch.linalg.matrix_norm(identity - cov_x, ord='fro')
    r2 = torch.linalg.matrix_norm(identity - cov_y, ord='fro')
    return (r1 + r2).mean()


def regularization_2(cov_x, cov_y, rank):
    r1 = rank + torch.trace(cov_x @ torch.log(cov_x) - cov_x)
    r2 = rank + torch.trace(cov_y @ torch.log(cov_y) - cov_y)
    return (r1 + r2).mean()
