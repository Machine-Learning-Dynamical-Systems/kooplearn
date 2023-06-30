from kooplearn._src import primal, dual

from kooplearn.data.datasets import mock_trajectory

def test_reduced_rank():
    X, Y = mock_trajectory(num_features=100)
    dual.fit_reduced_rank_regression_tikhonov
    dual.fit_rand_reduced_rank_regression_tikhonov
    primal.fit_rand_reduced_rank_regression_tikhonov
    primal.fit_reduced_rank_regression_tikhonov
    