from copy import deepcopy

import numpy as np

from kooplearn._src.utils import ShapeError
from kooplearn.data import TensorContextDataset


def parse_observables(
    inference_data: TensorContextDataset, data_fit: TensorContextDataset
):
    if inference_data.context_length != data_fit.context_length:
        raise ShapeError(
            f"The  context length ({inference_data.context_length}) of the inference data does not match the context length of the training data ({data_fit.context_length})."
        )
    lookback_len = inference_data.context_length - 1
    X_inference = inference_data.lookback(lookback_len)
    X_fit = data_fit.lookback(lookback_len)

    observables_dict = deepcopy(data_fit._observables_data)
    observables_dict["__state__"] = data_fit.data

    parsed_observables = {}
    observables_shapes = {}

    for obs_name, obs in observables_dict.items():
        if obs.dtype.kind != "f":
            raise TypeError(
                f"Observables should have floating-point values, whereas {obs_name} if of dtype {obs.dtype}"
            )
        obs = obs[:, -1]
        # If the observables are multidimensional, flatten them and save the shape for the final reshape
        trailing_dims = obs.shape[1:]
        expected_shape = (X_inference.shape[0],) + trailing_dims
        if obs.ndim > 2:
            obs = obs.reshape(
                obs.shape[0], -1
            )  # Flatten out everything for proper broadcasting
        parsed_observables[obs_name] = obs
        observables_shapes[obs_name] = expected_shape

    return parsed_observables, observables_shapes, X_inference, X_fit


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
