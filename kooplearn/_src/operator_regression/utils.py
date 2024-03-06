import numpy as np

from kooplearn._src.utils import ShapeError
from kooplearn.data import ContextWindow


def parse_observables(data: ContextWindow, data_fit: ContextWindow, observables_dict):
    if data.context_length != data_fit.context_length:
        raise ShapeError(
            f"The  context length ({data.context_length}) of the validation data does not match the context length of the training data ({data_fit.context_length})."
        )
    lookback_len = data.context_length - 1
    X_inference = data.lookback(lookback_len)
    X_fit, Y_fit = data.lookback(lookback_len), data.lookforward(lookback_len)

    if observables_dict is None:
        observables_dict = {"__state__": Y_fit}
    else:
        observables_dict["__state__":Y_fit]

    parsed_obs = {}
    expected_shapes = {}
    for obs_name, obs in observables_dict.keys():
        if callable(obs):
            obs = np.asanyarray(obs(observables_dict["__state__"]))
        else:
            try:
                obs = np.asanyarray(obs)
            except Exception as _:
                raise ValueError("Observables must be either, or callable.")

        if obs.dtype.kind != "f":
            raise TypeError(
                f"Observables should have floating-point values, whereas {obs_name} if of dtype {obs.dtype}"
            )
        if obs.shape[0] != observables_dict["__state__"].shape[0]:
            raise ShapeError(
                f"The observable {obs_name} was evaluated for {obs.shape[0]}, while the fitting data have {observables_dict['__state__'].shape[0]} examples."
            )
        # Reshape the observables to 2D arrays
        if obs.ndim == 1:
            obs = obs[:, None]

        # If the observables are multidimensional, flatten them and save the shape for the final reshape
        trailing_dims = obs.shape[1:]
        expected_shape = (X_inference.shape[0],) + trailing_dims
        if obs.ndim > 2:
            obs = obs.reshape(
                obs.shape[0], -1
            )  # Flatten out everything for proper broadcasting
        parsed_obs[obs_name] = obs
        expected_shapes[obs_name] = expected_shape

    return parsed_obs, expected_shapes, X_inference, X_fit


# !! Possibly to deprecate
def contexts_to_markov_train_states(
    contexts: ContextWindow,
    # lookback_len: int,
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
    if not (contexts.shape[1] == contexts._lookback_len + 1):
        raise ShapeError(
            f"This function act on context windows with lookforward dimension == 1, while context_len = {contexts.shape[1]} and lookback_len = {contexts._lookback_len}, that is lookforward_len = {contexts.shape[1] - contexts._lookback_len} != 1"
        )

    _init = contexts.lookback(0).data
    _evolution = contexts.lookback(1).data
    return _init, _evolution


def contexts_to_markov_predict_states(
    contexts,
    # lookback_len: int,
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

    if contexts._lookback_len == contexts.shape[1]:
        X = contexts.data
        Y = None
    elif contexts._lookback_len + 1 == contexts.shape[1]:
        X = contexts.lookback(0).data
        Y = contexts.lookforward().data
    else:
        raise ShapeError(
            f"This function act on context windows with lookforward dimension == 0 or 1, while context_len = {contexts.shape[1]} and lookback_len = {contexts._lookback_len}, that is lookforward_len = {contexts.shape[1] - contexts._lookback_len} != 0 or 1"
        )
    return X, Y
