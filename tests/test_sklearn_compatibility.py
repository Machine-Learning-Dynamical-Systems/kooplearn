from kooplearn.kernel._base import Kernel
from kooplearn.kernel._nystroem import NystroemKernel
from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([Kernel(), NystroemKernel()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)