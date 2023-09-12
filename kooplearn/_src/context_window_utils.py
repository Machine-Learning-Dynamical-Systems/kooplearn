import numpy as np
import logging
logger = logging.getLogger('kooplearn')

def check_contexts(
    data: np.ndarray,
    lookback_len: int,
    enforce_len1_lookforward: bool = False,
    warn_len0_lookforward: bool = False,
):
    data = np.asanyarray(data)
    assert isinstance(lookback_len, int), f"The lookback_len must be an int, while {type(lookback_len)=}"

    if data.ndim < 2:
        raise ValueError(f'Invalid shape {data.shape}. The data must have at least two dimensions [n_samples, context_len].')  
    if data.ndim == 2:
        logger.warn(f'The data has two axes, with an overall shape {data.shape}. The first axis is the time axis, whereas the second axis is the context axis. Appending a new axis to the end to represent (1-dimensional) features.')
        data = data[:, :, np.newaxis]
    if enforce_len1_lookforward and (lookback_len != data.shape[1] - 1):
        raise ValueError(f'The lookforward window has length {data.shape[1] - lookback_len}, but it should be of length 1.')
    if warn_len0_lookforward and lookback_len == data.shape[1]:
        logger.warn("The data has no lookforward window. This means that it can only be used for inference, but not for training.")
    if lookback_len > data.shape[1]:
        raise ValueError(f'Invalid lookback_len={lookback_len} for data of shape {data.shape}.')
    if lookback_len < 1:
        raise ValueError(f'Invalid lookback_len={lookback_len}.')
    return data

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

def contexts_to_markov_train_states(
    contexts: np.ndarray,
    lookback_len: int,
) -> np.ndarray:
    """TODO: Docstring for contexts_to_markov_IO_states.

    Args:
        contexts (np.ndarray): Array of contexts with shape ``(n_samples, context_len, *features_shape)``
        lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None.

        .. caution::

        If the lookforward window is larger than 1, ``context_len - lookback_len - 1`` an error will be raised.

    Returns:
        tuple(np.ndarray, np.ndarray): TODO.
    """
    
    contexts = check_contexts(contexts, lookback_len=lookback_len, enforce_len1_lookforward=True)

    _init = contexts[:, :-1, ...]
    _evolution = contexts[:, 1:, ...]
    return _init, _evolution

def contexts_to_markov_predict_states(
    contexts: np.ndarray,
    lookback_len: int,
) -> np.ndarray:
    """TODO: Docstring for contexts_to_markov_IO_states.

    Args:
        contexts (np.ndarray): Array of contexts with shape ``(n_samples, context_len, *features_shape)``
        lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None.

        .. caution::

        If the lookforward window is larger than 1, ``context_len - lookback_len - 1`` an error will be raised.

    Returns:
        tuple(np.ndarray, np.ndarray): TODO.
    """

    contexts = check_contexts(contexts, lookback_len=lookback_len)
    if lookback_len == contexts.shape[1]:
        X = contexts
        Y = None
    elif lookback_len + 1 == contexts.shape[1]:
        X = contexts[:, :-1, ...]
        Y = contexts[:, -1, ...]
    else:
        raise ValueError(f'Invalid lookback_len={lookback_len} for contexts of shape {contexts.shape}.') 
    return X, Y