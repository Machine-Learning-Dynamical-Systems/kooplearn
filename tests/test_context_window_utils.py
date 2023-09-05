import numpy as np
from kooplearn._src.context_window_utils import trajectory_to_contexts, stack_lookback, unstack_lookback
import pytest

@pytest.mark.parametrize('context_window_len', [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize('time_lag', [-1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize('trj_len', [1, 2, 10, 50])
@pytest.mark.parametrize('n_feature_axes', [0, 1, 2, 3])
def test_traj_to_contexts(context_window_len, time_lag, trj_len, n_feature_axes):
    trj = np.arange(trj_len)
    if n_feature_axes > 0:
        trj = np.reshape(trj, (trj_len,) + (1,)*n_feature_axes)
        assert trj.shape == (trj_len,) + (1,)*n_feature_axes
    else:
        assert trj.shape == (trj_len,)
        n_feature_axes = 1
    _C = 1 + (context_window_len - 1)*time_lag

    if context_window_len < 2:
        with pytest.raises(ValueError):
            _ = trajectory_to_contexts(trj, context_window_len, time_lag)
        return

    if time_lag < 1:
        with pytest.raises(ValueError):
            _ = trajectory_to_contexts(trj, context_window_len, time_lag)
        return

    if _C > trj_len:
        with pytest.raises(ValueError):
            _ = trajectory_to_contexts(trj, context_window_len, time_lag)
        return
    
    res = trajectory_to_contexts(trj, context_window_len, time_lag)
    assert res.shape == (trj_len -_C + 1, context_window_len) + (1,)*n_feature_axes
    if n_feature_axes == 1:
        for i in range(res.shape[1]):
            assert np.all(res[:, i, 0] == np.arange(trj_len -_C + 1) + i*time_lag)

@pytest.mark.parametrize('context_window_len', [2, 3, 4])
@pytest.mark.parametrize('lookback_len', [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize('trj_len', [1, 2, 10, 50])
@pytest.mark.parametrize('n_feature_axes', [0, 1, 2, 3])
def test_stack_lookback(context_window_len, lookback_len, trj_len, n_feature_axes):
    trj = np.arange(context_window_len)
    if n_feature_axes > 0:
        trj = np.reshape(trj, (context_window_len,) + (1,)*n_feature_axes)
    if n_feature_axes == 0 and trj_len == 1:
        pass
    else:    
        trj = np.stack([trj]*trj_len, axis=0)
    
    if lookback_len is not None:
        if lookback_len < 1:
            with pytest.raises(ValueError):
                _ = stack_lookback(trj, lookback_len)
            return
        
        if lookback_len >= context_window_len:
            with pytest.raises(ValueError):
                _ = stack_lookback(trj, lookback_len)
            return

    res = stack_lookback(trj, lookback_len)
    if lookback_len is None:
        lookback_len = context_window_len - 1
    assert res.shape == (trj_len, 2, lookback_len) + (1,)*n_feature_axes
    if n_feature_axes == 0:
        _r = res[0]
        assert np.all(_r[0] == np.arange(lookback_len))
        assert np.all(_r[1] == np.arange(lookback_len) + 1)
    if n_feature_axes == 0 and trj_len == 1:
        pass
    else:    
        assert np.all(trj[:, :lookback_len + 1, ...] == unstack_lookback(res))
