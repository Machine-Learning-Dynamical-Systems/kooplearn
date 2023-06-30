from typing import Optional
import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import eigsh
from kooplearn._src.utils import topk, weighted_norm

def fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank:int, tikhonov_reg:float, svd_solver: str = 'arnoldi'):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY@(C_XY.T) 
    if svd_solver == 'arnoldi':
        #Adding a small buffer to the Arnoldi-computed eigenvalues.
        values, vectors = eigsh(_crcov, rank + 3, M = reg_input_covariance)
    else:
        values, vectors = eigh(_crcov, reg_input_covariance)
    
    top_eigs = topk(values, rank)
    vectors = vectors[:, top_eigs.indices]
    values = top_eigs.values

    _norms = weighted_norm(vectors, reg_input_covariance)
    vectors = vectors@np.diag(_norms**(-1.0))
    return vectors

def fit_rand_reduced_rank_regression_tikhonov(
        C_X, 
        C_XY, 
        rank:int, 
        tikhonov_reg:float, 
        n_oversamples:int, 
        iterated_power:int,
        rng_seed: int = 0):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY@(C_XY.T) 
    rng = np.random.default_rng(rng_seed)
    sketch = rng.standard_normal(size = (reg_input_covariance.shape[0], rank + n_oversamples))

    for _ in range(iterated_power):
        _tmp_sketch = solve(reg_input_covariance, sketch, assume_a='pos')
        sketch = _crcov@_tmp_sketch


    sketch_p = solve(reg_input_covariance, sketch, assume_a='pos')  

    F_0 = (sketch_p.T)@sketch
    F_1 = (sketch_p.T)@(_crcov@sketch_p)
    
    values, vectors = eigh(F_1, F_0)
    _norms = weighted_norm(vectors, F_0)
    vectors = vectors@np.diag(_norms**(-1.0))
    return sketch_p@vectors[:, topk(values, rank).indices]

def fit_tikhonov(C_X, tikhonov_reg:float, rank: Optional[int] = None, svd_solver:str = 'arnoldi'):
    dim = C_X.shape[0]
    if rank is None:
        rank = dim
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    reg_input_covariance = C_X + tikhonov_reg*np.identity(dim, dtype=C_X.dtype)
    if svd_solver == 'arnoldi':
        values, vectors = eigsh(reg_input_covariance, k = rank, which = 'LM')
    else:
        values, vectors = eigh(reg_input_covariance)
    
    top_eigs = topk(values, rank)
    vectors = vectors[:, top_eigs.indices]
    values = top_eigs.values

    rsqrt_evals = np.diag(values**(-0.5))
    return vectors@rsqrt_evals

def fit_rand_tikhonov(C_X, C_XY, rank:int, tikhonov_reg:float, n_oversamples:int, iterated_power:int):
    raise NotImplementedError

def low_rank_predict(num_steps: int, U, C_XY, phi_testX, phi_trainX, obs_train_Y):
    # G = U U.T C_XY
    # G^n = (U)(U.T C_XY U)^(n-1)(U.T C_XY)
    num_train = phi_trainX.shape[0]
    phi_testX_dot_U = phi_testX@U
    U_C_XY_U = np.linalg.multi_dot([U.T, C_XY, U])
    U_phi_X_obs_Y = np.linalg.multi_dot([U.T, phi_trainX.T, obs_train_Y])*(num_train**-1)
    M = np.linalg.matrix_power(U_C_XY_U, num_steps-1)
    return np.linalg.multi_dot([phi_testX_dot_U, M, U_phi_X_obs_Y])

def low_rank_eig(U, C_XY):
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, vectors = eig(M)
    return values, U@vectors

def low_rank_eigfun_eval(phi_testX, vectors):
    return phi_testX@vectors

def svdvals(U, C_XY):
    M = np.linalg.multi_dot([U, U.T, C_XY])
    return np.linalg.svd(M, compute_uv=False)
                                 