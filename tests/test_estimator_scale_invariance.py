import numpy as np
import pytest

from kooplearn._src.operator_regression import dual, primal
from kooplearn.datasets import Mock


@pytest.mark.parametrize("scale_factor", [0.5, 2])
@pytest.mark.parametrize("tikhonov_reg", [0.0, 1e-3])
@pytest.mark.parametrize("svd_solver", ["full", "arnoldi"])
def test_reduced_rank_tikhonov_primal_scale_invariance(
    svd_solver, tikhonov_reg, scale_factor
):
    num_features = 10
    num_test_pts = 100
    rank = 5

    dataset = Mock(num_features=num_features, rng_seed=42)
    _Z = dataset.sample(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rdim = np.true_divide(1, X.shape[0])
    C_X = rdim * ((X.T) @ X)
    C_XY = rdim * ((X.T) @ Y)

    # Primal
    U = primal.fit_reduced_rank_regression(
        C_X, C_XY, tikhonov_reg, rank, svd_solver=svd_solver
    )
    U_scaled = primal.fit_reduced_rank_regression(
        scale_factor * C_X,
        scale_factor * C_XY,
        tikhonov_reg * scale_factor,
        rank,
        svd_solver=svd_solver,
    )

    G = np.linalg.multi_dot([U, U.T, C_XY])
    G_scaled = np.linalg.multi_dot([U_scaled, U_scaled.T, scale_factor * C_XY])
    assert np.allclose(G, G_scaled)


@pytest.mark.parametrize("scale_factor", [0.5, 2.0])
@pytest.mark.parametrize("tikhonov_reg", [1e-3])
@pytest.mark.parametrize("svd_solver", ["full", "arnoldi"])
def test_reduced_rank_tikhonov_dual_scale_invariance(
    svd_solver, tikhonov_reg, scale_factor
):
    num_features = 10
    num_test_pts = 100
    rank = 5

    dataset = Mock(num_features=num_features, rng_seed=42)
    _Z = dataset.sample(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    K_X = X @ (X.T)
    K_Y = Y @ (Y.T)

    # Dual
    U, V, _ = dual.fit_reduced_rank_regression(
        K_X, K_Y, tikhonov_reg, rank, svd_solver=svd_solver
    )
    U_scaled, V_scaled, _ = dual.fit_reduced_rank_regression(
        scale_factor * K_X,
        scale_factor * K_Y,
        tikhonov_reg * scale_factor,
        rank,
        svd_solver=svd_solver,
    )

    G = np.linalg.multi_dot([U, V.T])
    G_scaled = np.linalg.multi_dot([U_scaled, V_scaled.T])
    assert np.allclose(G * (scale_factor**-1), G_scaled, atol=1e-5)
