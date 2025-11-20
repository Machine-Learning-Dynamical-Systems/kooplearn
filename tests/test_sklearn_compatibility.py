from sklearn.utils.estimator_checks import parametrize_with_checks

from kooplearn.kernel._base import KernelRidge
from kooplearn.kernel._nystroem import NystroemKernelRidge
from kooplearn.linear_model._base import Ridge

MSG = "Data is sampled from a Gaussian distribution with mean 100 leading to numerical instabilities."
EXPECTED_FAILED_CHECKS = {
    "NystroemKernelRidge": {
        "check_fit_idempotent": MSG,
        "check_fit_check_is_fitted": MSG,
        "check_n_features_in": MSG,
    }
}

@parametrize_with_checks(
    [KernelRidge(), NystroemKernelRidge(), Ridge()],
    expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(est.__class__.__name__, {})
)
def test_sklearn_compatibility(estimator, check):
    check(estimator)
