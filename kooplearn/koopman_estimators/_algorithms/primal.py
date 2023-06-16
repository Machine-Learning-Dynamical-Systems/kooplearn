from typing import Optional
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh, lstsq, solve
from scipy.sparse.linalg import eigs, eigsh, cg, lsqr
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv
from sklearn.utils.extmath import randomized_svd
from .utils import topk, weighted_norm

def fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank:int, tikhonov_reg:float):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY@(C_XY.T) 

    values, vectors = eigh(_crcov, reg_input_covariance)
    _norms = weighted_norm(vectors, reg_input_covariance)
    vectors = vectors@np.diag(_norms**(-1.0))
    return vectors[:, topk(values, rank).indices]

def fit_rand_reduced_rank_regression_tikhonov(
        C_X, 
        C_XY, 
        rank:int, 
        tikhonov_reg:float, 
        n_oversamples:int, 
        iterated_power:int):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY@(C_XY.T) 
    sketch = np.random.normal(size = (reg_input_covariance.shape[0], rank + n_oversamples))

    for _ in range(iterated_power):
        _tmp_sketch = solve(reg_input_covariance, sketch, assume_a='pos')
        sketch = _crcov@_tmp_sketch

    sketch_p =  solve(reg_input_covariance, sketch, assume_a='pos')  
    F_0 = (sketch_p.T)@sketch
    F_1 = (sketch_p.T)@(_crcov@sketch_p)
    
    values, vectors = eigh(F_1, F_0)
    _norms = weighted_norm(vectors, F_0)
    vectors = vectors@np.diag(_norms**(-1.0))
    return vectors[:, topk(values, rank).indices]


def _postprocess_tikhonov_fit(S, V, rank:int, dim:int, rcond:float):
    pass

def fit_tikhonov(C_X, tikhonov_reg:float, rank: Optional[int] = None, svd_solver:str = 'arnoldi', rcond:float = 2.2e-16):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    Lambda, Q = eigh(reg_input_covariance)
    rsqrt_Lambda = np.diag(Lambda**(-0.5))
    return Q@rsqrt_Lambda

def low_rank_predict(num_steps: int, U, V, K_YX, K_testX, obs_train_Y):
    pass

def low_rank_eig(U, C_XY):
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, vectors = eig(M)
    return values, U@vectors

def low_rank_eigfun_eval(phi_testX, vectors):
    return phi_testX@vectors