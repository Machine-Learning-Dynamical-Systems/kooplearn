from sklearn.utils.estimator_checks import parametrize_with_checks
from sys import path
path.append('../')
from kooplearn.kernels import Kernel
from kooplearn.estimators import ReducedRank, PrincipalComponent

import numpy as np

class PaddedLinearKernel(Kernel):
    """Dummy kernel for testing. If X and Y have the same number of features is a linear kernel. Otherwise the smaller of the two is padded with zeros and the other is linearly combined with the padded one."""
    def __init__(self):
        pass
    def __call__(self, X, Y=None, backend='numpy'):
        if Y is None:
            lin_X = np.dot(X, X.T)
            return lin_X
        if Y is not None:
            if Y.shape[1] == X.shape[1]:
                pass
            elif X.shape[1] < Y.shape[1]:
                _zeroes = np.zeros((X.shape[0], Y.shape[1] - X.shape[1]))
                X = np.c_[X, _zeroes]  
            else:
                _zeroes = np.zeros((X.shape[0], X.shape[1] - Y.shape[1]))
                Y = np.c_[Y, _zeroes]
            lin_XY = np.dot(X, Y.T)
            return lin_XY
                
kernel = PaddedLinearKernel()
parameters = {
    'kernel': kernel,
    'svd_solver': 'full',
    'rank': 1
}
tikhonov_reg = None
@parametrize_with_checks([ReducedRank(**parameters, tikhonov_reg=tikhonov_reg), PrincipalComponent(**parameters)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)