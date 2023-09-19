import numpy as np
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