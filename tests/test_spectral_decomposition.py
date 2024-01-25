import numpy as np
import pytest

from kooplearn._src.operator_regression import dual, primal
from kooplearn.datasets import Mock


def _primal_right_normalization(right_vectors: np.ndarray):
    _norms = np.sum(right_vectors.conj() * right_vectors, axis=0)
    return np.allclose(_norms, np.ones(_norms.shape[0]))


def _primal_eigenvalue_equation(
    eigenvalues: np.ndarray,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    estimator: np.ndarray,
):
    reconstruction = np.linalg.multi_dot(
        [right_vectors, np.diag(eigenvalues), left_vectors.T]
    )
    return np.allclose(estimator, reconstruction)


def _primal_biortogonality(left_vectors: np.ndarray, right_vectors: np.ndarray):
    return np.allclose((left_vectors.T) @ right_vectors, np.eye(left_vectors.shape[1]))


@pytest.mark.parametrize("tikhonov_reg", [1e-3])
@pytest.mark.parametrize("svd_solver", ["full", "arnoldi"])
def test_primal_eig_decomposition(tikhonov_reg, svd_solver):
    num_features = 20
    num_test_pts = 200
    rank = 5

    dataset = Mock(num_features=num_features, rng_seed=42)
    _Z = dataset.sample(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rdim = np.true_divide(1, X.shape[0])
    C_X = rdim * ((X.T) @ X)
    C_XY = rdim * ((X.T) @ Y)

    U = primal.fit_principal_component_regression(
        C_X, tikhonov_reg, rank=rank, svd_solver=svd_solver
    )
    eig, lv, rv = primal.estimator_eig(U, C_XY)
    estimator = np.linalg.multi_dot([U, U.T, C_XY])

    assert _primal_right_normalization(rv)
    assert _primal_eigenvalue_equation(eig, lv, rv, estimator)
    assert _primal_biortogonality(lv, rv)


def _dual_right_normalization(right_vectors: np.ndarray, K_X: np.ndarray):
    r_dim = (K_X.shape[0]) ** (-1)
    _norms = np.sum(right_vectors.conj() * (r_dim * K_X @ right_vectors), axis=0)
    return np.allclose(_norms, np.ones(_norms.shape[0]))


def _dual_biortogonality(
    left_vectors: np.ndarray, right_vectors: np.ndarray, K_YX: np.ndarray
):
    r_dim = (K_YX.shape[0]) ** (-1)
    return np.allclose(
        np.linalg.multi_dot([left_vectors.T, r_dim * K_YX, right_vectors]),
        np.eye(left_vectors.shape[1]),
    )


@pytest.mark.parametrize("tikhonov_reg", [1e-3])
@pytest.mark.parametrize("svd_solver", ["full", "arnoldi"])
def test_dual_eig_decomposition(tikhonov_reg, svd_solver):
    num_features = 20
    num_test_pts = 200
    rank = 5

    dataset = Mock(num_features=num_features, rng_seed=42)
    _Z = dataset.sample(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    K_X = X @ (X.T)
    K_YX = Y @ (X.T)

    # Dual
    U, V, _ = dual.fit_principal_component_regression(
        K_X, tikhonov_reg, rank=rank, svd_solver=svd_solver
    )
    eig, lv, rv = dual.estimator_eig(U, V, K_X, K_YX)

    assert _dual_right_normalization(rv, K_X)
    assert _dual_biortogonality(lv, rv, K_YX)
