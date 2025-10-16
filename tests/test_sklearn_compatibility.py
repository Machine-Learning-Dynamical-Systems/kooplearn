from sklearn.utils.estimator_checks import parametrize_with_checks

from kooplearn.kernel._base import KernelRidge
from kooplearn.kernel._nystroem import NystroemKernelRidge
from kooplearn.linear_model._base import Ridge


@parametrize_with_checks([KernelRidge(), NystroemKernelRidge(), Ridge()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)
