import numpy as np
from typing import Optional
import logging
logger = logging.getLogger('kooplearn')

def trajectory_to_contexts(
    trajectory: np.ndarray,
    context_window_len: int,
    time_lag: int = 1
):
    """
    Working notes: the first axis is always assumed to be the time axis.
    """
    if context_window_len < 2:
        raise ValueError(f'context_window_len must be >= 2, got {context_window_len}')
    
    if time_lag < 1:
        raise ValueError(f'time_lag must be >= 1, got {time_lag}')
    
    trajectory = np.asanyarray(trajectory)
    if trajectory.ndim == 0:
        trajectory = trajectory.reshape(1, 1)
    elif trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]
    
    _context_window_len = 1 + (context_window_len - 1)*time_lag
    if _context_window_len > trajectory.shape[0]:
        raise ValueError(f'Invalid combination of context_window_len={context_window_len} and time_lag={time_lag} for trajectory of length {trajectory.shape[0]}. Try reducing context_window_len or time_lag.')
    
    _res =  np.lib.stride_tricks.sliding_window_view(trajectory, _context_window_len, axis=0)
    _res = np.moveaxis(_res, -1, 1)[:, ::time_lag, ...]
    return _res

def stack_lookback(
    contexts: np.ndarray,
    lookback_len: Optional[int] = None
) -> np.ndarray:
    """_summary_

    Args:
        contexts (np.ndarray): Array of contexts with shape ``(n_samples, len_context, *features_shape)``
        lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None.

        .. caution::

        If the lookforward window is larger than 1, ``len_context - lookback_len - 1`` snapshots will be discarted for each context window.

    Returns:
        np.ndarray: Array of length 2 contexts where the lookback windows of the input arrays are stacked on axis 2. The shape of the output array is therefore ``(n_samples, 2, lookback_len, *features_shape)``.
    """
    contexts = np.asanyarray(contexts)
    assert contexts.ndim >= 2
    if lookback_len is None:
        lookback_len = contexts.shape[1] - 1
    
    if lookback_len > contexts.shape[1] - 1:
        raise ValueError(f'Invalid lookback_len={lookback_len} for contexts of shape {contexts.shape}.')
    elif lookback_len < 1:
        raise ValueError(f'Invalid lookback_len={lookback_len}.')
    elif lookback_len < contexts.shape[1] - 1:
        logger.warn(f"The lookforward window ({contexts.shape[1] - lookback_len}) is larger than 1. For each context window, {contexts.shape[1] - lookback_len - 1} snapshots will be discarted.")
    _ctx = contexts[:, :lookback_len + 1, ...]
    _in = _ctx[:, :-1, ...]
    _out = _ctx[:, 1:, ...]
    return np.stack((_in, _out), axis=1)

    