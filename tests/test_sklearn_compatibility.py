from kooplearn.kernel._base import Kernel
from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([Kernel()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)