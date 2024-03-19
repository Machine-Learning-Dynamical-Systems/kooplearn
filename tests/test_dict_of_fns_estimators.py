from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
from scipy.stats import special_ortho_group

from kooplearn.abc import FeatureMap
from kooplearn.data import traj_to_contexts
from kooplearn.datasets.stochastic import LinearModel
from kooplearn.models.dict_of_fns import Nonlinear
from kooplearn.models.feature_maps import IdentityFeatureMap

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


class PolyFeatureMap(FeatureMap):
    def __init__(self, max_degree: int = 2):
        self._n = max_degree

    def __call__(self, X):
        _X = np.power(X[..., None], np.arange(1, self._n + 1))
        return _X.reshape(X.shape[0], -1)


@pytest.mark.parametrize("feature_map", [IdentityFeatureMap(), PolyFeatureMap()])
@pytest.mark.parametrize("reduced_rank", [True, False])
@pytest.mark.parametrize("rank", [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize("tikhonov_reg", [None, 0.0, 1e-10, 1e-5])
@pytest.mark.parametrize("svd_solver", ["full", "arnoldi", "wrong"])
@pytest.mark.parametrize("observables", [None, {"zeroes": np.zeros, "ones": np.ones}])
@pytest.mark.parametrize("predict_observables", [True, False])
@pytest.mark.parametrize("lookback_len", [1, 2, 3])
def test_Nonlinear_fit_predict_eig_modes_save_load(
    feature_map,
    reduced_rank,
    rank,
    tikhonov_reg,
    svd_solver,
    observables,
    predict_observables,
    lookback_len,
):
    dataset = make_linear_system()
    _Z = dataset.sample(np.zeros(DIM), NUM_SAMPLES)
    data = traj_to_contexts(_Z, lookback_len + 1)
    if svd_solver not in ["full", "arnoldi"]:
        with pytest.raises(ValueError):
            model = Nonlinear(
                feature_map=feature_map,
                reduced_rank=reduced_rank,
                rank=rank,
                tikhonov_reg=tikhonov_reg,
                svd_solver=svd_solver,
            )
    else:
        model = Nonlinear(
            feature_map=feature_map,
            reduced_rank=reduced_rank,
            rank=rank,
            tikhonov_reg=tikhonov_reg,
            svd_solver=svd_solver,
        )
        assert model.is_fitted is False
        model.fit(data)
        assert model.is_fitted is True

        if (observables is None) or (predict_observables is False):
            data.observables = observables
            X_pred = model.predict(data, predict_observables=predict_observables)
            assert X_pred.shape == data.lookforward(model.lookback_len).shape
            modes, _ = model.modes(data, predict_observables=predict_observables)
            assert (
                modes.shape
                == (model.rank,) + data.lookforward(model.lookback_len).shape
            )
        else:
            obs_shape = (len(data), 1, 2, 3, 4)
            data.observables = {
                k: v(obs_shape, dtype=np.float_) for k, v in observables.items()
            }

            X_pred = model.predict(data, predict_observables=predict_observables)
            assert "__state__" in X_pred.keys()
            for k, v in X_pred.items():
                if k == "__state__":
                    assert v.shape == data.lookforward(model.lookback_len).shape
                else:
                    assert v.shape == obs_shape

            modes, _ = model.modes(data, predict_observables=predict_observables)
            assert "__state__" in modes.keys()
            for k, v in modes.items():
                if k == "__state__":
                    assert (
                        v.shape
                        == (model.rank,) + data.lookforward(model.lookback_len).shape
                    )
                else:
                    assert v.shape == (model.rank,) + obs_shape

        vals, lv, rv = model.eig(eval_left_on=data, eval_right_on=data)
        assert vals.shape[0] <= rank
        assert vals.ndim == 1
        tmp_path = Path(__file__).parent / "tmp/model.bin"
        model.save(tmp_path)
        restored_model = Nonlinear.load(tmp_path)
        assert np.allclose(model.cov_X, restored_model.cov_X)
        assert np.allclose(model.cov_Y, restored_model.cov_Y)
        assert np.allclose(model.cov_XY, restored_model.cov_XY)
        assert np.allclose(model.data_fit, restored_model.data_fit)
        assert np.allclose(model.lookback_len, model.lookback_len)
        rmtree(Path(__file__).parent / "tmp/")
