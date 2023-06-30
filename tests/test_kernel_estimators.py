import pytest
from kooplearn._src.kernels import Linear, RBF, Matern
from kooplearn.data.datasets import mock_trajectory
from kooplearn.models import KernelDMD, KernelReducedRank



@pytest.mark.parametrize('kernel', [Linear(), RBF(), Matern()])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi', 'randomized'])
@pytest.mark.parametrize('tikhonov_reg', [None, 1e-3])
def test_kernelDMD(kernel, svd_solver, tikhonov_reg):
    num_features = 10
    num_samples = 100

    X, Y = mock_trajectory(num_features=num_features, num_samples=num_samples)

    model = KernelDMD(kernel=kernel, tikhonov_reg=tikhonov_reg, svd_solver=svd_solver)

    model.fit(X, Y)
    model.eig(eval_left_on = X, eval_right_on= X)
    model.predict(X)

@pytest.mark.parametrize('kernel', [Linear(), RBF(), Matern()])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi', 'randomized'])
@pytest.mark.parametrize('tikhonov_reg', [None, 1e-3])
def test_kernelReducedRank(kernel, svd_solver, tikhonov_reg):
    num_features = 10
    num_samples = 100

    X, Y = mock_trajectory(num_features=num_features, num_samples=num_samples)

    model = KernelReducedRank(kernel=kernel, tikhonov_reg=tikhonov_reg, svd_solver=svd_solver)

    model.fit(X, Y)
    model.eig(eval_left_on = X, eval_right_on= X)
    model.predict(X)
