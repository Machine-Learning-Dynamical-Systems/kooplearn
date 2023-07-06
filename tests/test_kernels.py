import pytest
import numpy as np
from kooplearn._src import kernels
from kooplearn.data.datasets import MockData

@pytest.mark.parametrize('size_x', [0, 1, 3])
@pytest.mark.parametrize('size_y', [-1, 0, 1, 3])
@pytest.mark.parametrize('kernel', [kernels.RBF(), kernels.ExpSineSquared(), kernels.Matern(), kernels.Poly(), kernels.Linear(), kernels.Quadratic()])
def test_predefined_kernels(size_x, size_y, kernel):
    num_features = 1
    dataset = MockData(num_features = num_features, rng_seed = 42)
    print(kernel)

    if type(kernel) in [kernels.Poly, kernels.Linear, kernels.Quadratic]:
        assert kernel.is_inf_dimensional == False
    else:
        assert kernel.is_inf_dimensional == True

    if size_x == 0:
        X = 0.5
        shape_x = 1
    elif size_x == 1:
        X = np.ones(10)
        shape_x = 10
    else:
        X = dataset.generate(None, size_x)
        shape_x = size_x + 1

    if size_y < 0:
        Y = None
        shape_y = shape_x
    elif size_y == 0:
        Y = 0.5
        shape_y = 1
    elif size_y == 1:
        Y = np.ones(10)
        shape_y = 10
    else:
        Y = dataset.generate(None, size_y)
        shape_y = size_y + 1
    _K = kernel(X, Y)
    assert _K.shape == (shape_x, shape_y)