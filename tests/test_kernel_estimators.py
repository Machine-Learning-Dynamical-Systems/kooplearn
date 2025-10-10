from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
from scipy.stats import special_ortho_group

from kooplearn.datasets.stochastic import LinearModel
from kooplearn.kernel import KernelRidge

from sklearn.utils.validation import check_is_fitted

TRUE_RANK = 5
DIM = 20
NUM_SAMPLES = 50


def make_linear_system():
    eigs = 9 * np.logspace(-3, -1, TRUE_RANK)
    # print("Eigenvalues:")
    # for ev in eigs:
    #     print(f"{ev:.1e} \t", end='')
    eigs = np.concatenate([eigs, np.zeros(DIM - TRUE_RANK)])
    Q = special_ortho_group(DIM, 0).rvs(1)
    A = np.linalg.multi_dot([Q, np.diag(eigs), Q.T])

    # Consistency-check
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(eigs))
    return LinearModel(A, noise=1e-5, rng_seed=0)


@pytest.mark.parametrize("kernel", ['linear', 'rbf', 'laplacian'])
@pytest.mark.parametrize("reduced_rank", [True, False])
@pytest.mark.parametrize("n_components", [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize("eigen_solver", ["auto","dense", "arpack", "randomized"])
@pytest.mark.parametrize("alpha", [None, 0.0, 1e-5])
@pytest.mark.parametrize("observables", [None, np.zeros, np.ones])
def test_Kernel_fit_predict_eig_modes_save_load(
    n_components,
    reduced_rank,
    kernel,
    alpha,
    eigen_solver,
    observables,
):
    dataset = make_linear_system()
    data = dataset.sample(np.zeros(DIM), NUM_SAMPLES)

    model = KernelRidge(
        n_components=n_components,
        reduced_rank=reduced_rank,
        kernel=kernel,
        alpha=alpha,
        eigen_solver=eigen_solver,
    )

    with pytest.raises(Exception):
        check_is_fitted(model)
    if reduced_rank and eigen_solver == "randomized" and (alpha is None or alpha == 0.0):
        with pytest.raises(ValueError):
            model.fit(data)
    else:
        model.fit(data)
        assert check_is_fitted(model) is None

        if (observables is None):
            X_pred = model.predict(data)
            assert X_pred.shape == model.X_fit_.shape
            modes, _ = model.modes(data)
            assert (
                modes.shape
                == (model.rank_,) + model.X_fit_.shape
            )
        else:
            obs_shape = (len(data), 1, 2, 3, 4)
            X_pred = model.predict(data, observable=observables(obs_shape))
            assert X_pred.shape == obs_shape

            modes, _ = model.modes(data, observable=observables(obs_shape))
            assert modes.shape == (model.rank_,) + obs_shape

        vals, lv, rv = model.eig(eval_left_on=data, eval_right_on=data)
        assert vals.shape[0] <= n_components
        assert vals.ndim == 1