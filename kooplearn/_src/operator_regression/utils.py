from kooplearn._src.context_window_utils import check_contexts, contexts_to_markov_predict_states
import numpy as np

def _parse_DMD_observables(observables, data, data_fit, lookback_len):
    #Shape checks:
    data = check_contexts(data, lookback_len,)
    if not ((data.shape[1] == data_fit.shape[1]) or (data.shape[1] == lookback_len)):
        raise ValueError(f"Shape mismatch between training data and inference data. The inference data has context length {data.shape[1]}, while the training data has context length {data_fit.shape[1]}.")

    X_inference, _ = contexts_to_markov_predict_states(data, lookback_len)
    X_fit, Y_fit = contexts_to_markov_predict_states(data_fit, lookback_len)

    if observables is None:
        _obs = Y_fit
    elif callable(observables):
        _obs = observables(Y_fit)
    elif isinstance(observables, np.ndarray):
        assert observables.shape[0] == Y_fit.shape[0], f"Observables have {observables.shape[0]} samples while the number of training data is {Y_fit.shape[0]}."
        _obs = observables
    else:
        raise ValueError(
            "Observables must be either None, a callable or a Numpy array of the observable evaluated at the training data points.")

    #Reshape the observables to 2D arrays
    if _obs.ndim == 1:
        _obs = _obs[:, None]

    #If the observables are multidimensional, flatten them and save the shape for the final reshape
    _obs_trailing_dims = _obs.shape[1:]
    expected_shape = (X_inference.shape[0],) + _obs_trailing_dims
    if _obs.ndim > 2:
        _obs = _obs.reshape(_obs.shape[0], -1)
    return _obs, expected_shape, X_inference, X_fit