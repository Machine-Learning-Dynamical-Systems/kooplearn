import pytest
import numpy as np
from numpy.typing import ArrayLike
from kooplearn.data.datasets import MockData
from kooplearn._src.operator_regression import primal, dual

def _primal_right_normalization(right_vectors: ArrayLike):
    _norms = np.sum(right_vectors.conj()*right_vectors, axis = 0)
    return np.allclose(_norms, np.ones(_norms.shape[0]))

def _primal_eigenvalue_equation(eigenvalues: ArrayLike, left_vectors: ArrayLike, right_vectors: ArrayLike, estimator: ArrayLike):
    reconstruction = np.linalg.multi_dot([right_vectors, np.diag(eigenvalues), left_vectors.T])
    return np.allclose(estimator, reconstruction)

def _primal_biortogonality(left_vectors: ArrayLike, right_vectors: ArrayLike):
    return np.allclose((left_vectors.T)@right_vectors, np.eye(left_vectors.shape[1]))

@pytest.mark.parametrize('tikhonov_reg', [1e-3])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi'])
def test_primal_eig_decomposition(tikhonov_reg, svd_solver):
    num_features = 20
    num_test_pts = 200
    rank = 5
    
    dataset = MockData(num_features = num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    
    rdim = np.true_divide(1, X.shape[0])
    C_X = rdim*((X.T)@X)
    C_XY = rdim*((X.T)@Y)

    U = primal.fit_tikhonov(C_X, tikhonov_reg, rank = rank, svd_solver = svd_solver)
    eig, lv, rv = primal.estimator_eig(U, C_XY)
    estimator = np.linalg.multi_dot([U, U.T, C_XY])

    assert _primal_right_normalization(rv)
    assert _primal_eigenvalue_equation(eig, lv, rv, estimator)
    assert _primal_biortogonality(lv, rv)