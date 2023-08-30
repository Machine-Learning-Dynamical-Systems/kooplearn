import torch


def dpnets_loss(x_encoded: torch.Tensor, y_encoded: torch.Tensor, p_loss_coef: float = 1.0, s_loss_coef: float = 0,
                reg_1_coef: float = 0, reg_2_coef: float = 0, rank: int = None):
    """Computes the loss used in [1].

    Parameters:
        x_encoded: value of x after being encoded.
        y_encoded: value of y after being encoded.
        p_loss_coef: Coefficient of the score function P.
        s_loss_coef: Coefficient of the score function S.
        reg_1_coef: Coefficient of the regularization term 1.
        reg_2_coef: Coefficient of the regularization term 2.
        rank: Rank of the Koopman estimator (only needed when using the regularization with term 2).

    [1] Vladimir Kostic, Pietro Novelli, Riccardo Grazzi, Karim Lounici, and Massimiliano Pontil. “Deep
    Projection Networks for Learning Time-Homogeneous Dynamical Systems.” arXiv, July 19,
    2023. https://doi.org/10.48550/arXiv.2307.09912.
    """
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
    r1 = rank + torch.trace(cov_x @ torch.log(cov_x.abs()) - cov_x)
    r2 = rank + torch.trace(cov_y @ torch.log(cov_y.abs()) - cov_y)
    return (r1 + r2).mean()
