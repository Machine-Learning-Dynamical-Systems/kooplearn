from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import eigsh
from kooplearn._src.utils import topk, weighted_norm

def fit_reduced_rank_regression_tikhonov(
        C_X: ArrayLike, #Input covariance matrix
        C_XY: ArrayLike, #Cross-covariance matrix
        tikhonov_reg:float, #Tikhonov regularization parameter, can be 0.0
        rank:int, #Rank of the estimator
        svd_solver: str = 'arnoldi' #SVD solver to use. Arnoldi is faster for low ranks.
    ):
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
        C_X: ArrayLike, #Input covariance matrix
        C_XY: ArrayLike, #Cross-covariance matrix
        tikhonov_reg:float, #Tikhonov regularization parameter
        rank:int, #Rank of the estimator
        n_oversamples:int, #Number of oversamples
        iterated_power:int, #Number of power iterations
        rng_seed: Optional[int] = None #Random seed
    ):
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

def fit_tikhonov(
        C_X: ArrayLike, #Input covariance matrix
        tikhonov_reg:float, #Tikhonov regularization parameter, can be 0
        rank: Optional[int] = None, #Rank of the estimator
        svd_solver:str = 'arnoldi' #SVD solver to use. Arnoldi is faster for low ranks.
    ):
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

def fit_rand_tikhonov(
        C_X: ArrayLike, #Input covariance matrix
        C_XY: ArrayLike, #Cross-covariance matrix
        tikhonov_reg:float, #Tikhonov regularization parameter
        rank:int, #Rank of the estimator
        n_oversamples:int, #Number of oversamples
        iterated_power:int, #Number of power iterations
        rng_seed:Optional[int] = None #Random seed
    ):
    raise NotImplementedError

def predict(
        num_steps: int, #Number of steps to predict (return the last one)
        U: ArrayLike, #Projection matrix, as returned by the fit functions defined above 
        C_XY: ArrayLike, #Cross-covariance matrix
        phi_Xin: ArrayLike, #Feature map evaluated on the initial conditions
        phi_X: ArrayLike, #Feature map evaluated on the training input data
        obs_train_Y: ArrayLike #Observable to be predicted evaluated on the output training data
    ):
    # G = U U.T C_XY
    # G^n = (U)(U.T C_XY U)^(n-1)(U.T C_XY)
    num_train = phi_X.shape[0]
    phi_Xin_dot_U = phi_Xin@U
    U_C_XY_U = np.linalg.multi_dot([U.T, C_XY, U])
    U_phi_X_obs_Y = np.linalg.multi_dot([U.T, phi_X.T, obs_train_Y])*(num_train**-1)
    M = np.linalg.matrix_power(U_C_XY_U, num_steps-1)
    return np.linalg.multi_dot([phi_Xin_dot_U, M, U_phi_X_obs_Y])

def estimator_eig(
        U: ArrayLike, #Projection matrix, as returned by the fit functions defined above 
        C_XY: ArrayLike, #Cross-covariance matrix
    ):
    #Using the trick described in https://arxiv.org/abs/1905.11490
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = eig(M, left = True, right = True)

    r_perm = np.argsort(values)
    l_perm = np.argsort(values.conj())
    values = values[r_perm]
    
    #Normalization in RKHS norm
    rv = U@rv
    rv = rv[:, r_perm]
    rv = rv/np.linalg.norm(rv, axis = 0)
    #Biorthogonalization
    lv = np.linalg.multi_dot([C_XY.T, U, lv])
    lv = lv[:, l_perm]
    l_norm = np.sum(lv*rv, axis=0)
    lv = lv/l_norm

    return values, lv, rv

def evaluate_eigenfunction(
        phi_Xin: ArrayLike, #Feature map evaluated on the initial conditions
        lv_or_rv: ArrayLike, #Left or right eigenvector, as returned by estimator_eig
    ):
    return phi_Xin@lv_or_rv

def svdvals(U, C_XY):
    M = np.linalg.multi_dot([U, U.T, C_XY])
    return np.linalg.svd(M, compute_uv=False)                         