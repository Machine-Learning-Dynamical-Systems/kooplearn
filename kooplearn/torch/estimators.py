from typing import Optional
import torch
import logging
from einops import einsum
from kooplearn.torch.typing import LinalgDecomposition, RealLinalgDecomposition
from kooplearn.torch.linalg import spd_norm, generalized_eigh

def reduced_rank_regression(
    input_covariance: torch.Tensor,
    cross_covariance: torch.Tensor, #C_{XY}
    tikhonov_reg: float,
    ) -> tuple:
    
    n = input_covariance.shape[0]
    reg_input_covariance = input_covariance + tikhonov_reg*torch.eye(n, dtype=input_covariance.dtype, device=input_covariance.device)

    _crcov = torch.mm(cross_covariance, cross_covariance.T)
    _values, _vectors = generalized_eigh(_crcov, reg_input_covariance) 
    
    _norms = spd_norm(_vectors, reg_input_covariance)
    vectors = _vectors*(1/_norms)
    return RealLinalgDecomposition(_values, vectors)

def tikhonov_regression(
    input_covariance: torch.Tensor,
    tikhonov_reg: float,
    ) -> tuple:
    n = input_covariance.shape[0]
    reg_input_covariance = input_covariance + tikhonov_reg*torch.eye(n, dtype=input_covariance.dtype, device=input_covariance.device)

    Lambda, Q = torch.linalg.eigh(reg_input_covariance)
    return RealLinalgDecomposition(Lambda, Q@torch.diag(Lambda.rsqrt()))

def eig(
    fitted_estimator: RealLinalgDecomposition, 
    cross_covariance: torch.Tensor,
    rank: Optional[int] = None,
    ) -> tuple:
    if rank is not None:
        _, idxs = torch.topk(fitted_estimator.values, rank)
        U = (fitted_estimator.vectors)[:, idxs]
    else:
        U = fitted_estimator.vectors
    
    #U@(U.T)@Tw = v w -> (U.T)@T@Uq = vq and w = Uq 
    values, Q = torch.linalg.eig((U.T)@(cross_covariance@U))
    dtype = torch.promote_types(Q.dtype, U.dtype)
    U = U.to(dtype)
    Q = Q.to(dtype)
    return LinalgDecomposition(values, torch.mm(U, Q))


def sq_error(
    input_data: torch.Tensor,
    output_data: torch.Tensor,
    estimator: torch.Tensor
    ) -> float:
    """Mean squared error between the output data and the estimator applied to the input data.

    Args:
        input_data (Tensor): Tensor of shape (n, dim) containing the input data.
        output_data (Tensor): Tensor of shape (n, dim) containing the output data.
        estimator (Tensor): Tensor of shape (dim, dim) containing the estimator.

    Returns:
        float: mean squared error between the output data and the estimator applied to the input data.
    """    
    err = output_data.T - torch.mm(estimator.T, input_data.T)
    return torch.mean(torch.linalg.vector_norm(err, ord=2, dim=0)**2)

def naive_predict(
    featurized_x: torch.Tensor,
    fitted_estimator: RealLinalgDecomposition, 
    input_data: torch.Tensor,
    output_raw_data: torch.Tensor,
    rank: Optional[int] = None,
    ) -> torch.Tensor:
    logging.warn("This function is deprecated. Use predict instead.")
    if rank is not None:
        _, idxs = torch.topk(fitted_estimator.values, rank)
        U = (fitted_estimator.vectors)[:, idxs]
    else:
        U = fitted_estimator.vectors

    x = einsum(
        featurized_x,
        U,
        U.T,
        input_data,
        output_raw_data,
        "d_in, d_in r, r d_out, n d_out, n dim -> dim"
    )
    num_data = float(input_data.shape[0])
    return (num_data**-1)*x

def predict(
    phi_X: torch.Tensor,
    fitted_estimator: RealLinalgDecomposition, 
    phi_training_X: torch.Tensor,
    phi_training_Y: torch.Tensor,
    training_Y: torch.Tensor,
    num_steps: int = 1,
    rank: Optional[int] = None,
) -> torch.Tensor:

    if rank is not None:
        _, idxs = torch.topk(fitted_estimator.values, rank)
        U = fitted_estimator.vectors[:, idxs]
    else:
        U = fitted_estimator.vectors

    phi_X_mul_U = torch.matmul(phi_X, U)
    num_data = float(training_Y.shape[0])
    phi_training_X_mul_training_Y = torch.reciprocal(num_data) * torch.matmul(phi_training_X.T, training_Y)
    training_cross_cov = torch.matmul(phi_training_X.T, phi_training_Y) * torch.reciprocal(float(phi_training_X.shape[0]))
    U_cross_U = torch.linalg.multi_dot([U.T, training_cross_cov, U])
    M = torch.linalg.matrix_power(U_cross_U, num_steps - 1)
    return torch.linalg.multi_dot([phi_X_mul_U, M, U.T, phi_training_X_mul_training_Y])
