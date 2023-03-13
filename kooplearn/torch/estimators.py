from typing import Optional
import torch
from torch import Tensor
from einops import einsum

def spd_norm(vecs: Tensor, spd_matrix: Tensor) -> Tensor:
     _v = torch.mm(spd_matrix, vecs)
     _v_T = torch.mm(spd_matrix.T, vecs)
     return torch.sqrt(0.5*torch.linalg.vecdot(vecs, _v + _v_T, dim = 0).real)

def reduced_rank_regression(
    input_covariance: Tensor,
    cross_covariance: Tensor, #C_{XY}
    tikhonov_reg: float,
    ) -> tuple:
    
    n = input_covariance.shape[0]
    reg_input_covariance = input_covariance + tikhonov_reg*torch.eye(n, dtype=input_covariance.dtype, device=input_covariance.device)

    _crcov = torch.mm(cross_covariance, cross_covariance.T)
    _values, _vectors = torch.lobpcg(_crcov, reg_input_covariance, n = n) 
    
    _norms = spd_norm(_vectors, reg_input_covariance)
    vectors = _vectors*(1/_norms)
    return _values, vectors

def tikhonov_regression(
    input_covariance: Tensor,
    tikhonov_reg: float,
    ) -> tuple:
    n = input_covariance.shape[0]
    reg_input_covariance = input_covariance + tikhonov_reg*torch.eye(n, dtype=input_covariance.dtype, device=input_covariance.device)

    Lambda, Q = torch.linalg.eigh(reg_input_covariance)
    return Lambda, Q@torch.diag(Lambda.rsqrt())

def eig(
    fitted_estimator: Tensor, 
    cross_covariance: Tensor,
    rank: Optional[int] = None,
    ) -> tuple:
      
    if rank is not None:
        _, idxs = torch.topk(fitted_estimator.values, rank)
        U = fitted_estimator[:, idxs]
    else:
        U = fitted_estimator
    
    #U@(U.T)@Tw = v w -> (U.T)@T@Uq = vq and w = Uq 
    values, Q = torch.linalg.eig((U.T)@(cross_covariance@U))
    return values, U@Q


def sq_error(
    input_data: Tensor,
    output_data: Tensor,
    estimator: Tensor
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
    featurized_x: Tensor,
    fitted_estimator: Tensor, 
    input_data: Tensor,
    output_raw_data: Tensor,
    rank: Optional[int] = None,
    ) -> Tensor:

    if rank is not None:
        _, idxs = torch.topk(fitted_estimator.values, rank)
        U = fitted_estimator[:, idxs]
    else:
        U = fitted_estimator

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