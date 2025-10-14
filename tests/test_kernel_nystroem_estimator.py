import numpy as np
import pytest
from scipy.stats import special_ortho_group
from sklearn.utils.validation import check_is_fitted

from kooplearn.datasets.stochastic import LinearModel
from kooplearn.kernel import NystroemKernelRidge

TRUE_RANK = 5
DIM = 20
NUM_SAMPLES = 50


def make_linear_system():
    eigs = 9 * np.logspace(-3, -1, TRUE_RANK)
    eigs = np.concatenate([eigs, np.zeros(DIM - TRUE_RANK)])
    Q = special_ortho_group(DIM, 0).rvs(1)
    A = np.linalg.multi_dot([Q, np.diag(eigs), Q.T])
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(eigs))
    return LinearModel(A, noise=1e-5, rng_seed=0)


@pytest.mark.parametrize("kernel", ['linear', 'rbf', 'laplacian'])
@pytest.mark.parametrize("lag_time", [1, 5])
@pytest.mark.parametrize("reduced_rank", [True, False])
@pytest.mark.parametrize("n_components", [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize("eigen_solver", ["auto", "dense", "arpack"])
@pytest.mark.parametrize("alpha", [None, 0.0, 1e-5])
@pytest.mark.parametrize("observables", [None, np.ones])
@pytest.mark.parametrize("tol", [0.0])
@pytest.mark.parametrize("max_iter", [None])
@pytest.mark.parametrize("n_centers", [0.1, 0.5, 50,])
@pytest.mark.parametrize("random_state", [None])
@pytest.mark.parametrize("copy_X", [True])
@pytest.mark.parametrize("n_jobs", [1])
def test_Kernel_fit_predict_eig_modes_risk_svals(
    n_components,
    lag_time,
    reduced_rank,
    kernel,
    alpha,
    eigen_solver,
    observables,
    tol,
    max_iter,
    n_centers,
    random_state,
    copy_X,
    n_jobs,
):
    dataset = make_linear_system()
    data = dataset.sample(np.zeros(DIM), NUM_SAMPLES)

    model = NystroemKernelRidge(
        n_components=n_components,
        lag_time=lag_time,
        reduced_rank=reduced_rank,
        kernel=kernel,
        alpha=alpha,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        n_centers=n_centers,
        random_state=random_state,
        copy_X=copy_X,
        n_jobs=n_jobs,
    )

    # model should not be fitted yet
    with pytest.raises(Exception):
        check_is_fitted(model)

    # randomized solver without regularization should fail
    if reduced_rank and eigen_solver == "randomized" and (alpha is None or alpha == 0.0):
        with pytest.raises(ValueError):
            model.fit(data)
        return

    model.fit(data)
    assert check_is_fitted(model) is None
    assert model.U_.shape[1] == model.rank_

    # Predict and modes checks
    if observables is None:
        X_pred = model.predict(data)
        assert X_pred.shape == model.X_fit_.shape
        modes, _ = model.modes(data)
        assert modes.shape == (model.rank_,) + model.X_fit_.shape
    else:
        obs_shape = (len(data), 1, 2, 3, 4)
        X_pred = model.predict(data, observable=observables(obs_shape))
        assert X_pred.shape == obs_shape
        modes, _ = model.modes(data, observable=observables(obs_shape))
        assert modes.shape == (model.rank_,) + obs_shape

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
        X_pred_2 = model.predict(data)
    else:
        X_pred_2 = model.predict(data, observable=observables(obs_shape))
    np.testing.assert_allclose(X_pred, X_pred_2, rtol=1e-10)


# ---- Extra edge case tests integrated below ----

def test_invalid_input_shape_raises():
    X_too_small = np.random.randn(1, 5)
    model = NystroemKernelRidge()
    with pytest.raises(ValueError):
        model.fit(X_too_small)


def test_predict_observable_shape_mismatch_raises():
    X = np.random.randn(20, 4)
    model = NystroemKernelRidge(kernel="linear")
    model.fit(X)
    wrong_obs = np.random.randn(X.shape[0] + 2, X.shape[1])
    with pytest.raises(ValueError):
        model.predict(X, observable=wrong_obs)


def test_callable_kernel_functionality():
    X = np.random.randn(15, 5)

    def custom_kernel(X, Y=None):
        # Ensure both X and Y are 2D
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        # Compute squared Euclidean distance
        dists = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)
        return np.exp(-0.1 * dists)

    model = NystroemKernelRidge(kernel=custom_kernel, n_components=3)
    model.fit(X)
    out = model.predict(X)
    assert out.shape == X.shape
    svals = model.svals()
    assert np.all(np.isfinite(svals))