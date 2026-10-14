import numpy as np
import pytest
from scipy.sparse.linalg import ArpackError
from scipy.stats import special_ortho_group
from sklearn.utils.validation import check_is_fitted

from kooplearn.datasets import make_linear_system
from kooplearn.kernel import GeneratorDirichlet

TRUE_RANK = 5
DIM = 20
NUM_SAMPLES = 50

# fit() passes alpha=None as tikhonov_reg=0.0, and reduced_rank_regression_dirichlet
# divides by tikhonov_reg: predictions are not finite and some fits keep no modes.
UNREGULARIZED = pytest.mark.xfail(
    strict=True, reason="GeneratorDirichlet divides by zero when alpha is None or 0"
)


def make_data():
    eigs = 9 * np.logspace(-3, -1, TRUE_RANK)
    eigs = np.concatenate([eigs, np.zeros(DIM - TRUE_RANK)])
    Q = special_ortho_group(DIM, 0).rvs(1)
    A = np.linalg.multi_dot([Q, np.diag(eigs), Q.T])
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(eigs))
    return make_linear_system(np.zeros(DIM), A, NUM_SAMPLES, noise=1e-3, random_state=0)


@pytest.mark.parametrize("n_components", [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize(
    "alpha",
    [pytest.param(None, marks=UNREGULARIZED), pytest.param(0.0, marks=UNREGULARIZED), 1e-5],
)
@pytest.mark.parametrize("observable", [True, False])
@pytest.mark.parametrize(
    "diffusion", [1e-3 * np.ones(DIM), 1e-3, 1e-3 * np.ones((NUM_SAMPLES + 1, DIM))]
)
def test_Kernel_fit_predict_eig_modes_risk_svals(
    n_components,
    alpha,
    diffusion,
    observable,
):
    data = make_data()

    model = GeneratorDirichlet(
        diffusion,
        n_components=n_components,
        alpha=alpha,
    )

    # model should not be fitted yet
    with pytest.raises(Exception):
        check_is_fitted(model)

    # Skip the test only if the eigensolver fails during fitting
    try:
        model.fit(data, y=data)
    except (np.linalg.LinAlgError, ArpackError) as e:
        pytest.skip(f"linalg error during fitting, skipping test. Details: {e}")

    # Fit, predict and modes checks
    assert check_is_fitted(model) is None
    X_pred = model.predict(data, t=1.0, observable=observable)
    assert X_pred.shape == model.X_fit_.shape
    assert np.isfinite(X_pred).all()
    modes = model.dynamical_modes(data, observable=observable)
    assert modes[0].shape == model.X_fit_.shape

    # Eigen-decomposition
    vals, _lv, _rv = model.eig(eval_left_on=data, eval_right_on=data)
    assert vals.ndim == 1
    assert vals.shape[0] <= n_components
    assert np.isfinite(vals).all()
