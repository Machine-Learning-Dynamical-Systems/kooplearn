import numpy as np
import pytest
from scipy.stats import special_ortho_group
from sklearn.utils.validation import check_is_fitted

from kooplearn.datasets import make_linear_system
from kooplearn.linear_model import Ridge

TRUE_RANK = 5
DIM = 20
NUM_SAMPLES = 50


def make_data():
    eigs = 9 * np.logspace(-3, -1, TRUE_RANK)
    eigs = np.concatenate([eigs, np.zeros(DIM - TRUE_RANK)])
    Q = special_ortho_group(DIM, 0).rvs(1)
    A = np.linalg.multi_dot([Q, np.diag(eigs), Q.T])
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(eigs))
    return make_linear_system(np.zeros(DIM), A, NUM_SAMPLES, noise=1e-3, random_state=0)


@pytest.mark.parametrize("reduced_rank", [True, False])
@pytest.mark.parametrize("lag_time", [1, 5])
@pytest.mark.parametrize("n_components", [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize("eigen_solver", ["auto", "dense", "arpack", "randomized"])
@pytest.mark.parametrize("alpha", [None, 0.0, 1e-6])
@pytest.mark.parametrize("observables", [None, np.zeros, np.ones])
@pytest.mark.parametrize("tol", [0.0])
@pytest.mark.parametrize("max_iter", [None])
@pytest.mark.parametrize("iterated_power", [5])
@pytest.mark.parametrize("n_oversamples", [5])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize("copy_X", [True])
def test_Kernel_fit_predict_eig_modes_risk_svals(
    n_components,
    lag_time,
    reduced_rank,
    alpha,
    eigen_solver,
    observables,
    tol,
    max_iter,
    iterated_power,
    n_oversamples,
    random_state,
    copy_X,
):
    data = make_data()

    model = Ridge(
        n_components=n_components,
        lag_time=lag_time,
        reduced_rank=reduced_rank,
        alpha=alpha,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        iterated_power=iterated_power,
        n_oversamples=n_oversamples,
        random_state=random_state,
        copy_X=copy_X,
    )

    # model should not be fitted yet
    with pytest.raises(Exception):
        check_is_fitted(model)

    # randomized solver without regularization should fail
    if (
        reduced_rank
        and eigen_solver == "randomized"
        and (alpha is None or alpha == 0.0)
    ):
        with pytest.raises(ValueError):
            model.fit(data)
        return

    model.fit(data)
    assert check_is_fitted(model) is None
    assert model.U_.shape[1] == model.rank_

    # Fit, predict and modes checks
    if observables is None:
        model.fit(data, y=observables)
        assert check_is_fitted(model) is None
        assert model.U_.shape[1] == model.rank_
        X_pred = model.predict(data, observable=False)
        assert X_pred.shape == model.X_fit_.shape
        modes = model.dynamical_modes(data, observable=False)
        assert modes[0].shape == model.X_fit_.shape
    else:
        obs_shape = (len(data), 6)
        model.fit(data, y=observables(obs_shape))
        assert check_is_fitted(model) is None
        assert model.U_.shape[1] == model.rank_
        X_pred = model.predict(data, observable=True)
        assert X_pred.shape == obs_shape
        modes = model.dynamical_modes(data, observable=True)
        assert modes[0].shape == obs_shape

    # Eigen-decomposition
    vals, lv, rv = model.eig(eval_left_on=data, eval_right_on=data)
    assert vals.ndim == 1
    assert vals.shape[0] <= n_components
    assert np.isfinite(vals).all()

    # Risk and singular values
    svals = model.svals()
    risk = model.risk()
    assert np.all(np.isfinite(svals))
    assert isinstance(risk, float)

    # Internal consistency
    if observables is None:
        X_pred_2 = model.predict(data, observable=False)
    else:
        X_pred_2 = model.predict(data, observable=True)
    np.testing.assert_allclose(X_pred, X_pred_2, rtol=1e-10)


# ---- Extra edge case tests integrated below ----


def test_invalid_input_shape_raises():
    X_too_small = np.random.randn(1, 5)
    model = Ridge()
    with pytest.raises(ValueError):
        model.fit(X_too_small)


def test_predict_observable_shape_mismatch_raises():
    X = np.random.randn(20, 4)
    model = Ridge()
    model.fit(X)
    wrong_obs = np.random.randn(X.shape[0] + 2, X.shape[1])
    with pytest.raises(ValueError):
        model.predict(X, observable=wrong_obs)


def test_random_state_reproducibility():
    data = make_data()
    model1 = Ridge(
        n_components=3,
        reduced_rank=True,
        alpha=1e-3,
        eigen_solver="randomized",
        random_state=42,
    )
    model2 = Ridge(
        n_components=3,
        reduced_rank=True,
        alpha=1e-3,
        eigen_solver="randomized",
        random_state=42,
    )
    model1.fit(data)
    model2.fit(data)
    np.testing.assert_allclose(model1.U_, model2.U_, rtol=1e-10)
