import pytest
from typing import NamedTuple
import numpy as np
from kooplearn._src.operator_regression import primal, dual
from kooplearn.data.datasets import MockData

class EigenDecomposition(NamedTuple):
    values: np.ndarray
    left: np.ndarray
    right: np.ndarray

def _compare_evd(evd_1:EigenDecomposition, evd_2:EigenDecomposition) -> bool:
    evd_1_sort = np.argsort(evd_1.values)
    evd_2_sort = np.argsort(evd_2.values)
    assert np.allclose(evd_1.values[evd_1_sort], evd_2.values[evd_2_sort])
    assert np.allclose(evd_1.left[:,evd_1_sort], evd_2.left[:,evd_2_sort])
    assert np.allclose(evd_1.right[:,evd_1_sort], evd_2.right[:,evd_2_sort])
    return True

@pytest.mark.parametrize('tikhonov_reg', [0.0, 1e-3])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi'])
@pytest.mark.parametrize('dt', [1, 2, 3])
def test_reduced_rank_tikhonov_primal_dual_consistency(dt, svd_solver, tikhonov_reg):
    num_features = 10
    num_test_pts = 100
    rank = 5
    
    dataset = MockData(num_features = num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, tikhonov_reg, rank, svd_solver = svd_solver)

    dual_predict = dual.predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, dual_lv, dual_rv = dual.estimator_eig(U, V, K_X, K_Y, K_YX)

    evd_dual = EigenDecomposition(
        dual_eig,
        dual.evaluate_eigenfunction(X_test@(Y.T), V, dual_lv),
        dual.evaluate_eigenfunction(X_test@(X.T), U, dual_rv)
    )

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_reduced_rank_regression_tikhonov(C_X, C_XY, tikhonov_reg, rank, svd_solver = svd_solver)

    primal_predict = primal.predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, primal_lv, primal_rv = primal.estimator_eig(U, C_XY)

    evd_primal = EigenDecomposition(
        primal_eig,
        primal.evaluate_eigenfunction(X_test, primal_lv),
        primal.evaluate_eigenfunction(X_test, primal_rv)
    )

    assert dual_predict.shape == (num_test_pts, num_features)
    assert np.allclose(primal_predict, dual_predict)
    assert _compare_evd(evd_primal, evd_dual)

@pytest.mark.parametrize('tikhonov_reg', [0.0, 1e-3])
@pytest.mark.parametrize('rank', [5, None])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi'])
@pytest.mark.parametrize('dt', [1, 2, 3])
def test_tikhonov_primal_dual_consistency(dt, svd_solver, rank, tikhonov_reg):
    num_features = 10
    num_test_pts = 100

    dataset = MockData(num_features=num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V = dual.fit_tikhonov(K_X, tikhonov_reg, rank = rank, svd_solver = svd_solver)

    dual_predict = dual.predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, dual_lv, dual_rv = dual.estimator_eig(U, V, K_X, K_Y, K_YX)

    evd_dual = EigenDecomposition(
        dual_eig,
        dual.evaluate_eigenfunction(X_test@(Y.T), V, dual_lv),
        dual.evaluate_eigenfunction(X_test@(X.T), U, dual_rv)
    )

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_tikhonov(C_X, tikhonov_reg, rank = rank, svd_solver = svd_solver)

    primal_predict = primal.predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, primal_lv, primal_rv = primal.estimator_eig(U, C_XY)

    evd_primal = EigenDecomposition(
        primal_eig,
        primal.evaluate_eigenfunction(X_test, primal_lv),
        primal.evaluate_eigenfunction(X_test, primal_rv)
    )

    assert dual_predict.shape == (num_test_pts, num_features)

    assert np.allclose(primal_predict, dual_predict)
    if rank is not None:
        assert _compare_evd(evd_primal, evd_dual)

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank(dt):
    num_features = 50
    num_test_pts = 200
    rank = 3
    tikhonov_reg = 1e-3
    n_oversamples = 20
    iterated_power = 2

    dataset = MockData(num_features=num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_rand_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg, n_oversamples, False, iterated_power)

    dual_predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_rand_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg, n_oversamples, iterated_power)

    primal_predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, _ = primal.low_rank_eig(U, C_XY)

    assert dual_predict.shape == (num_test_pts, num_features)
    assert np.allclose(primal_predict, dual_predict)
    assert np.allclose(np.sort(dual_eig.real), np.sort(primal_eig.real))
    assert np.allclose(np.sort(dual_eig.imag), np.sort(primal_eig.imag))

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank_primal(dt):
    num_features = 50
    num_test_pts = 200
    rank = 5
    tikhonov_reg = 1e-3
    n_oversamples = 20
    iterated_power = 3

    dataset = MockData(num_features=num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]

    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    C_X = rdim*((X.T)@X)
    C_XY = rdim*((X.T)@Y)
    
    #Primal
    U = primal.fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg)

    predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    eig, _ = primal.low_rank_eig(U, C_XY)
    
    #Rand
    U_rand = primal.fit_rand_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg, n_oversamples, iterated_power)

    rand_predict = primal.low_rank_predict(dt, U_rand, C_XY, X_test, X, Y)
    rand_eig, _ = primal.low_rank_eig(U_rand, C_XY)

    diff = np.ravel(rand_predict - predict)
    mean = 0.5*np.ravel(rand_predict + predict)
    print(np.linalg.norm(diff, ord=np.inf)/np.linalg.norm(mean, ord = np.inf))

    assert predict.shape == (num_test_pts, num_features)
    assert rand_predict.shape == (num_test_pts, num_features)

    assert np.allclose(predict, rand_predict)
    assert np.allclose(np.sort(rand_eig.real), np.sort(eig.real))
    assert np.allclose(np.sort(rand_eig.imag), np.sort(eig.imag))

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank_dual(dt):
    num_features = 50
    num_test_pts = 200
    rank = 10
    tikhonov_reg = 1e-3
    n_oversamples = 10
    iterated_power = 2

    dataset = MockData(num_features=num_features, rng_seed = 42)
    _Z = dataset.generate(None, num_test_pts)
    X, Y = _Z[:-1], _Z[1:]
    
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    K_X = X@(X.T)
    K_Y = Y@(Y.T)
    K_YX = Y@(X.T)
    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg)

    predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    #Rand
    U_rand, V_rand, _ = dual.fit_rand_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg, n_oversamples, False, iterated_power)

    rand_predict = dual.low_rank_predict(dt, U_rand, V_rand, K_YX, K_testX, Y)
    rand_eig, _, _ = dual.low_rank_eig(U_rand, V_rand, K_X, K_Y, K_YX)

    assert predict.shape == (num_test_pts, num_features)
    assert rand_predict.shape == (num_test_pts, num_features)

    assert np.allclose(predict, rand_predict)
    assert np.allclose(np.sort(rand_eig.real), np.sort(eig.real))
    assert np.allclose(np.sort(rand_eig.imag), np.sort(eig.imag))