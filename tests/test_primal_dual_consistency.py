import pytest
import numpy as np
from kooplearn._src import primal, dual
from kooplearn.data.datasets import mock_trajectory

@pytest.mark.parametrize('dt', [1, 5, 10])
def test_reduced_rank_predict(dt):
    num_features = 50
    num_test_pts = 200
    rank = 10
    tikhonov_reg = 1e-3

    X, Y = mock_trajectory(num_features=num_features)
    X_test = np.random.rand(num_test_pts, num_features)
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)
    C_Y = rdim*((Y.T)@Y)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg)

    dual_predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, _, _ = dual.low_rank_eig(U, V, K_Y, K_YX)

    #Primal
    U = primal.fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg)

    primal_predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, _ = primal.low_rank_eig(U, C_XY)

    assert np.allclose(primal_predict, dual_predict)
    assert np.allclose(np.sort(dual_eig.real), np.sort(primal_eig.real))
    assert np.allclose(np.sort(dual_eig.imag), np.sort(primal_eig.imag))

