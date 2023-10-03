import numpy as np

from kooplearn._src.utils import ShapeError, check_contexts_shape


def parse_observables(observables, data, data_fit, lookback_len):
    # Shape checks:
    check_contexts_shape(data, lookback_len, is_inference_data=True)
    check_contexts_shape(data_fit, lookback_len)
    data = np.asanyarray(data)

    X_inference, _ = contexts_to_markov_predict_states(data, lookback_len)
    X_fit, Y_fit = contexts_to_markov_predict_states(data_fit, lookback_len)

    if observables is None:
        _obs = Y_fit
    elif callable(observables):
        _obs = observables(Y_fit)
    else:
        raise ValueError("Observables must be either None, or callable.")

    # Reshape the observables to 2D arrays
    if _obs.ndim == 1:
        _obs = _obs[:, None]

    # If the observables are multidimensional, flatten them and save the shape for the final reshape
    _obs_trailing_dims = _obs.shape[1:]
    expected_shape = (X_inference.shape[0],) + _obs_trailing_dims
    if _obs.ndim > 2:
        _obs = _obs.reshape(_obs.shape[0], -1)
    return _obs, expected_shape, X_inference, X_fit


def contexts_to_markov_train_states(
    contexts: np.ndarray,
    lookback_len: int,
) -> np.ndarray:
    """TODO: Docstring for contexts_to_markov_IO_states.

    Args:
        contexts (np.ndarray): Array of contexts with shape ``(n_samples, context_len, *features_shape)``
        lookback_len (int): Length of the lookback window associated to the contexts.

        .. caution::

        If the lookforward window ``context_len - lookback_len - 1`` is not equal to 1, an error will be raised.

    Returns:
        tuple(np.ndarray, np.ndarray): TODO.
    """
    if not (contexts.shape[1] == lookback_len + 1):
        raise ShapeError(
            f"This function act on context windows with lookforward dimension == 1, while context_len = {contexts.shape[1]} and lookback_len = {lookback_len}, that is lookforward_len = {contexts.shape[1] - lookback_len} != 1"
        )

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
        lookback_len (int): Length of the lookback window associated to the contexts.

        .. caution::

        If the lookforward window ``context_len - lookback_len - 1`` is larger than 1, an error will be raised.

    Returns:
        tuple(np.ndarray, np.ndarray): TODO.
    """

    if lookback_len == contexts.shape[1]:
        X = contexts
        Y = None
    elif lookback_len + 1 == contexts.shape[1]:
        X = contexts[:, :-1, ...]
        Y = contexts[:, -1, ...]
    else:
        raise ShapeError(
            f"This function act on context windows with lookforward dimension == 0 or 1, while context_len = {contexts.shape[1]} and lookback_len = {lookback_len}, that is lookforward_len = {contexts.shape[1] - lookback_len} != 0 or 1"
        )
    return X, Y
