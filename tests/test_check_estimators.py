from sklearn.utils.estimator_checks import parametrize_with_checks
from sys import path
path.append('../')
from kooplearn.kernels import Kernel
from kooplearn.sklearn_estimators import ReducedRankRegression, PrincipalComponentRegression

import numpy as np

class DummyKernel(Kernel):
    """Dummy kernel for testing. Accepting X and Y with different number of features."""
    def __init__(self):
        pass
    def __call__(self, X, Y=None, backend='numpy'):
        lin_X = np.dot(X, X.T)
        lin_X /= np.linalg.norm(lin_X, ord='fro')
        if Y is not None:
            lin_Y = np.dot(Y, Y.T)
            lin_Y /= np.linalg.norm(lin_Y, ord='fro')
        else:
            lin_Y = 0
        return lin_X + lin_Y
kernel = DummyKernel()
@parametrize_with_checks([ReducedRankRegression(kernel=kernel), PrincipalComponentRegression(kernel=kernel)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)