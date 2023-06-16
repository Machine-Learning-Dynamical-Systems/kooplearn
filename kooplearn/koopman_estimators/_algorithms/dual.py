from typing import Optional
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh, lstsq
from scipy.sparse.linalg import eigs, eigsh, cg, lsqr
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv
from sklearn.utils.extmath import randomized_svd
from .utils import topk, modified_QR, weighted_norm

def fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank:int, tikhonov_reg:float, svd_solver:str = 'arnoldi'):
    dim = K_X.shape[0]
    rsqrt_dim = dim**(-0.5)
    alpha = dim*tikhonov_reg
    #Rescaled Kernel matrices
    K_Xn = K_X*rsqrt_dim
    K_Yn = K_Y*rsqrt_dim

    K = K_Yn@K_Xn
    #Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow. Prefer svd_solver == 'randomized' in such a case.
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    if svd_solver == 'arnoldi':
        #Adding a small buffer to the Arnoldi-computed eigenvalues.
        sigma_sq, U = eigs(K, rank + 3, K_X + tikhonov)  
    else: #'full'     
        sigma_sq, U = eig(K, K_X + tikhonov)
    
    max_imag_part = np.max(U.imag)
    if max_imag_part >=2.2e-10:
        logging.warn(f"The computed projector is not real. The Kernel matrix is severely ill-conditioned.")
    U = np.real(U)

    #Post-process U. Promote numerical stability via additional QR decoposition if necessary.
    U = U[:, topk(sigma_sq.real, rank).indices]

    norm_inducing_op = (K_Xn@(K_Xn.T)) + tikhonov_reg*K_X
    U, _, columns_permutation = modified_QR(U, M = norm_inducing_op, column_pivoting=True)
    U = U[:,np.argsort(columns_permutation)]
    if U.shape[1] < rank:
        logging.warn(f"The numerical rank of the projector is smaller than the selected rank ({rank}). {rank - U.shape[1]} degrees of freedom will be ignored.")
        _zeroes = np.zeros((U.shape[0], rank - U.shape[1]))
        U = np.c_[U, _zeroes]
        assert U.shape[1] == rank
    V = K_X@np.asfortranarray(U)
    return U, V, sigma_sq

def fit_rand_reduced_rank_regression_tikhonov(
        K_X, 
        K_Y, 
        rank:int, 
        tikhonov_reg:float, 
        n_oversamples:int, 
        optimal_sketching:bool, 
        iterated_power:int):
    dim = K_X.shape[0]
    inv_dim = dim**(-1.0)
    alpha = dim*tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    K_reg_inv = IterInv(K_X + tikhonov, ifunc = cg)
    l = rank + n_oversamples
    if optimal_sketching:
        Cov = inv_dim*K_Y
        Om = np.random.multivariate_normal(np.zeros(dim, dtype=K_X.dtype), Cov, size=l).T
    else:
        Om = np.random.randn(dim, l)      
        
    for _ in range(iterated_power):
        #Powered randomized rangefinder
        Om = (inv_dim*K_Y)@(Om - alpha*K_reg_inv@Om)    
    KOm = K_reg_inv@Om
    KOmp = Om - alpha*KOm
    
    F_0 = (Om.T@KOmp)
    F_1 = (KOmp.T @ (inv_dim*(K_Y @ KOmp)))

    #Generation of matrices U and V.   
    try:
        sigma_sq, Q = eigh(F_1, F_0)
    except LinAlgError:  
        sigma_sq, Q = eig(pinvh(F_0)@F_1) 
    
    Q_norm = np.sum(Q.conj()*(F_0@Q), axis=0)
    Q = Q@np.diag(Q_norm**-0.5)
    _idxs = topk(sigma_sq.real, rank).indices
    sigma_sq = sigma_sq.real
    
    Q = Q[:,_idxs] 
    U = (dim**0.5)*np.asfortranarray(KOm @ Q)
    V = (dim**0.5)*np.asfortranarray(KOmp @ Q)
    return U.real, V.real, sigma_sq

def fit_reduced_rank_regression_noreg(K_X, K_Y, rank: int, svd_solver:str = 'arnoldi'):
    #Solve the Hermitian eigenvalue problem to find V
    if svd_solver != 'full':
        sigma_sq, V = eigsh(K_Y, rank)
    else:
        sigma_sq, V = eigh(K_Y)
        V = V[:, topk(sigma_sq, rank).indices]
    
    #Normalize V
    _V_norm = np.linalg.norm(V,ord=2,axis=0)/np.sqrt(V.shape[0])
    V = V@np.diag(_V_norm**-1)

    #Solve the least squares problem to determine U
    if svd_solver != 'full':
        U = np.zeros_like(V)
        for i in range(U.shape[1]):
            U[:,i] = lsqr(K_X, V[:,i])[0] #Not optimal with this explicit loop
    else:
        U = lstsq(K_X, V)[0]
    return U, V, sigma_sq


def _postprocess_tikhonov_fit(S, V, rank:int, dim:int, rcond:float):
    top_svals = topk(S, rank)   
    V = V[:,top_svals.indices]
    S = top_svals.values
    
    _test = S>rcond
    if all(_test):            
        V = np.sqrt(dim)*(V@np.diag(S**-0.5))
    else:
        V = np.sqrt(dim)*(V[:, _test]@np.diag(S[_test]**-0.5))
        logging.warn(f"The numerical rank of the projector is smaller than the selected rank ({rank}). {rank - V.shape[1]} degrees of freedom will be ignored.")
        _zeroes = np.zeros((V.shape[0], rank - V.shape[1]))
        V = np.c_[V, _zeroes]
        assert V.shape[1] == rank
    return S, V  

def fit_tikhonov(K_X, tikhonov_reg:float, rank: Optional[int] = None, svd_solver:str = 'arnoldi', rcond:float = 2.2e-16):
    dim = K_X.shape[0]
    if rank is None:
        rank = dim
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    alpha = dim*tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    if svd_solver == 'arnoldi':
        S, V = eigsh(K_X + tikhonov, rank)
    elif svd_solver == 'full':
        S, V = eigh(K_X + tikhonov)   
    S, V = _postprocess_tikhonov_fit(S, V, rank, dim, rcond)
    return V, V

def fit_rand_tikhonov(K_X, tikhonov_reg: float, rank: int, n_oversamples: int, iterated_power: int, rcond:float = 2.2e-16):
    dim = K_X.shape[0]
    alpha = dim*tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    V, S, _ = randomized_svd(K_X + tikhonov, rank, n_oversamples=n_oversamples, n_iter=iterated_power, random_state=None)
    S, V = _postprocess_tikhonov_fit(S, V, rank, dim, rcond)
    return V, V, S

def low_rank_predict(num_steps: int, U, V, K_YX, K_testX, obs_train_Y):
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    dim = U.shape[0]
    rsqrt_dim = dim**(-0.5)
    K_dot_U = rsqrt_dim*K_testX@U
    V_dot_obs = rsqrt_dim*(V.T)@obs_train_Y
    V_K_XY_U = (dim**-1)*np.linalg.multi_dot([V.T, K_YX, U])
    M = np.linalg.matrix_power(V_K_XY_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])

def low_rank_eig(U, V, K_X, K_Y, K_YX):
    r_dim = (K_X.shape[0])**(-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim*K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim*K_X, U])
    W_Y = np.linalg.multi_dot([V.T, r_dim*K_Y, V])

    w, vl, vr =  eig(W_YX, left=True, right=True) #Left -> V, Right -> U
    
    #Normalization
    norm_r = weighted_norm(vr,W_X) 
    norm_l = weighted_norm(vl,W_Y) 

    vr = vr @ np.diag(norm_r**(-1))
    vl = vl @ np.diag(norm_l**(-1))

    return w, vl, vr

def low_rank_eigfun_eval(K_testX, U_or_V, vr_or_vl):
    rsqrt_dim = (U_or_V.shape[0])**(-0.5)
    return np.linalg.multi_dot([rsqrt_dim*K_testX, U_or_V, vr_or_vl])

def svdvals(U, V, K_X, K_Y):
    #Inefficient implementation
    rdim = (K_X.shape[0])**(-1)
    A = np.linalg.multi_dot([V.T, rdim*K_Y, V])
    B = np.linalg.multi_dot([U.T, rdim*K_X, U])
    v = eig(A@B, left=False, right=False)
    #Clip the negative values
    v = v.real
    v[v<0] = 0
    return np.sqrt(v)