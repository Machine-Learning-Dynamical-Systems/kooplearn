import pytest
from kooplearn._src.kernels import Linear, RBF, Matern
from kooplearn.data.datasets import MockData
from kooplearn.models import KernelDMD, KernelReducedRank


@pytest.mark.parametrize('kernel', [Linear(), RBF(), Matern()])
@pytest.mark.parametrize('solver', ['full', 'arnoldi', 'randomized'])
@pytest.mark.parametrize('tikhonov_reg', [None, 1e-3])
def test_kernelDMD(kernel, solver, tikhonov_reg):
    num_features = 10
    num_samples = 100

    dataset = MockData(num_features=num_features, rng_seed=42)
    _Z = dataset.generate(None, num_samples)
    X, Y = _Z[:-1], _Z[1:]

    model = KernelDMD(kernel=kernel, tikhonov_reg=tikhonov_reg, solver=solver)

    model.fit(X, Y)
    model.eig(eval_left_on=X, eval_right_on=X)
    model.predict(X)


@pytest.mark.parametrize('kernel', [Linear(), RBF(), Matern()])
@pytest.mark.parametrize('solver', ['full', 'arnoldi', 'randomized'])
@pytest.mark.parametrize('tikhonov_reg', [None, 1e-3])
def test_kernelReducedRank(kernel, solver, tikhonov_reg):
    num_features = 10
    num_samples = 100

    dataset = MockData(num_features=num_features, rng_seed=42)
    _Z = dataset.generate(None, num_samples)
    X, Y = _Z[:-1], _Z[1:]
    model = KernelReducedRank(kernel=kernel, tikhonov_reg=tikhonov_reg, solver=solver)
    if (tikhonov_reg is None) and (solver == 'randomized'):
        with pytest.raises(ValueError):
            model.fit(X, Y)
    else:
        model.fit(X, Y)
        model.eig(eval_left_on=X, eval_right_on=X)
        model.predict(X)
