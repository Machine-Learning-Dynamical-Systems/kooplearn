from typing import Optional
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh, lstsq, solve
from scipy.sparse.linalg import eigs, eigsh, cg, lsqr
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv
from sklearn.utils.extmath import randomized_svd
from .utils import topk, modified_QR, weighted_norm

def fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank:int, tikhonov_reg:float, svd_solver:str = 'arnoldi'):
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
        optimal_sketching:bool, 
        iterated_power:int):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY@(C_XY.T) 
    sketch = np.random.normal(size = (reg_input_covariance.shape[0], rank + n_oversamples))

    for _ in range(iterated_power):
        _tmp_sketch = solve(reg_input_covariance, sketch, assume_a='pos')
        sketch = _crcov@_tmp_sketch

    sketch_p =  jax.scipy.linalg.solve(reg_input_covariance, sketch, assume_a='pos')  
    F_0 = (sketch_p.T)@sketch
    F_1 = (sketch_p.T)@(_crcov@sketch_p)
    
    _gep = generalized_eigh(F_1, F_0)
    _norms = batch_spd_norm(_gep.vectors, F_0)
    vectors = _gep.vectors*(jax.lax.reciprocal(_norms))

def fit_reduced_rank_regression_noreg(K_X, K_Y, rank: int, svd_solver:str = 'arnoldi'):
    pass


def _postprocess_tikhonov_fit(S, V, rank:int, dim:int, rcond:float):
    pass

def fit_tikhonov(K_X, rank: Optional[int] = None, svd_solver:str = 'arnoldi', rcond:float = 2.2e-16):
    pass

def fit_rand_tikhonov(K_X, rank: int, n_oversamples: int, iterated_power: int, rcond:float = 2.2e-16):
    pass

def low_rank_predict(num_steps: int, U, V, K_YX, K_testX, obs_train_Y):
    pass

def low_rank_eig(U, V, K_X, K_Y, K_YX):
    pass

def low_rank_eigfun_eval(K_testX, U_or_V, vr_or_vl):
    pass