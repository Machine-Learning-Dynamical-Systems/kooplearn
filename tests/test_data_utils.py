import numpy as np
import pytest

from kooplearn.data import traj_to_contexts


@pytest.mark.parametrize("backend", ["torch", "numpy", "auto"])
@pytest.mark.parametrize("context_window_len", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("time_lag", [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("trj_len", [1, 2, 10, 50])
@pytest.mark.parametrize("n_feature_axes", [0, 1, 2, 3])
def test_traj_to_contexts(
    backend, context_window_len, time_lag, trj_len, n_feature_axes
):
    trj = np.arange(trj_len)
    if n_feature_axes > 0:
        trj = np.reshape(trj, (trj_len,) + (1,) * n_feature_axes)
        assert trj.shape == (trj_len,) + (1,) * n_feature_axes
    else:
        assert trj.shape == (trj_len,)
        n_feature_axes = 1
    _C = 1 + (context_window_len - 1) * time_lag

    if context_window_len < 1:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(
                trj,
                context_window_len=context_window_len,
                time_lag=time_lag,
                backend=backend,
            )
        return

    if time_lag < 1:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(
                trj,
                context_window_len=context_window_len,
                time_lag=time_lag,
                backend=backend,
            )
        return

    if _C > trj_len:
        with pytest.raises(ValueError):
            _ = traj_to_contexts(
                trj,
                context_window_len=context_window_len,
                time_lag=time_lag,
                backend=backend,
            )
        return

    res = traj_to_contexts(trj, context_window_len, time_lag)
    assert res.shape == (trj_len - _C + 1, context_window_len) + (1,) * n_feature_axes
    if n_feature_axes == 1:
        for i in range(res.shape[1]):
            assert np.all(
                res.data[:, i, 0] == np.arange(trj_len - _C + 1) + i * time_lag
            )
