import numpy as np
import pytest

from kooplearn.data import traj_to_contexts
from kooplearn.nn.data import TrajToContextsDataset


@pytest.mark.parametrize("context_window_len", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("time_lag", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("trj_len", [1, 2, 10, 50])
@pytest.mark.parametrize("n_feature_axes", [0, 1, 2, 3])
def test_traj_to_contexts(context_window_len, time_lag, trj_len, n_feature_axes):
    trj = np.arange(trj_len)
    if n_feature_axes > 0:
        trj = np.reshape(trj, (trj_len,) + (1,) * n_feature_axes)
        assert trj.shape == (trj_len,) + (1,) * n_feature_axes
    else:
        assert trj.shape == (trj_len,)
        n_feature_axes = 1
    _C = 1 + (context_window_len - 1) * time_lag

    if context_window_len < 2:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(trj, context_window_len, time_lag)
        return

    if time_lag < 1:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(trj, context_window_len, time_lag)
        return

    if _C > trj_len:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(trj, context_window_len, time_lag)
        return

    res = traj_to_contexts(trj, context_window_len, time_lag)
    assert res.shape == (trj_len - _C + 1, context_window_len) + (1,) * n_feature_axes
    if n_feature_axes == 1:
        for i in range(res.shape[1]):
            assert np.all(res[:, i, 0] == np.arange(trj_len - _C + 1) + i * time_lag)


@pytest.mark.parametrize("context_window_len", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("time_lag", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("trj_len", [1, 2, 10, 50])
@pytest.mark.parametrize("n_feature_axes", [0, 1, 2, 3])
def test_torch_traj_to_contexts(context_window_len, time_lag, trj_len, n_feature_axes):
    trj = np.arange(trj_len)
    if n_feature_axes > 0:
        trj = np.reshape(trj, (trj_len,) + (1,) * n_feature_axes)
        assert trj.shape == (trj_len,) + (1,) * n_feature_axes
    else:
        assert trj.shape == (trj_len,)
        n_feature_axes = 1
    _C = 1 + (context_window_len - 1) * time_lag

    if context_window_len < 2:
        with pytest.raises(ValueError):
            _ = TrajToContextsDataset(trj, context_window_len, time_lag)
        return

    if time_lag < 1:
        with pytest.raises(ValueError):
            _ = TrajToContextsDataset(trj, context_window_len, time_lag)
        return

    if _C > trj_len:
        with pytest.raises(ValueError):
            _ = TrajToContextsDataset(trj, context_window_len, time_lag)
        return

    res = (
        TrajToContextsDataset(trj, context_window_len, time_lag)
        .contexts.detach()
        .cpu()
        .numpy()
    )
    assert res.shape == (trj_len - _C + 1, context_window_len) + (1,) * n_feature_axes
    if n_feature_axes == 1:
        for i in range(res.shape[1]):
            assert np.all(res[:, i, 0] == np.arange(trj_len - _C + 1) + i * time_lag)
