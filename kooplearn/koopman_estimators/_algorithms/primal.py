from typing import Optional
import logging
import numpy as np
from scipy.linalg import eig, eigh, LinAlgError, pinvh, lstsq
from scipy.sparse.linalg import eigs, eigsh, cg, lsqr
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv
from sklearn.utils.extmath import randomized_svd
from .utils import topk, modified_QR, weighted_norm

def fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank:int, tikhonov_reg:float, svd_solver:str = 'arnoldi'):
    pass

def fit_rand_reduced_rank_regression_tikhonov(
        K_X, 
        K_Y, 
        rank:int, 
        tikhonov_reg:float, 
        n_oversamples:int, 
        optimal_sketching:bool, 
        iterated_power:int):
    pass

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