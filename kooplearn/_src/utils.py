from typing import NamedTuple, Callable
import numpy as np
import os
import math
from pathlib import Path
from kooplearn._src.context_window_utils import check_contexts, contexts_to_markov_predict_states
from scipy.spatial.distance import pdist
import logging
logger = logging.getLogger('kooplearn')

class TopKReturnType(NamedTuple):
    values: np.ndarray
    indices: np.ndarray

def topk(vec: np.ndarray, k: int):
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = np.flip(np.argsort(vec))  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return TopKReturnType(values, indices)

def fuzzy_parse_complex(vec: np.ndarray, tol: float = 10.0):
    assert issubclass(vec.dtype.type, np.complexfloating), "The input element should be complex"
    rcond = tol*np.finfo(vec.dtype).eps
    pdist_real_part = pdist(vec.real[:, None])
    #Set the same element whenever pdist is smaller than eps*tol
    condensed_idxs = np.argwhere(pdist_real_part < rcond)[:, 0]
    fuzzy_real = vec.real.copy()
    if condensed_idxs.shape[0] >=1:  
        for idx in condensed_idxs:
            i, j = row_col_from_condensed_index(vec.real.shape[0], idx)
            avg = 0.5*(fuzzy_real[i] + fuzzy_real[j])
            fuzzy_real[i] = avg
            fuzzy_real[j] = avg
    fuzzy_imag = vec.imag.copy()
    fuzzy_imag[np.abs(fuzzy_imag)<rcond] = 0.0
    return fuzzy_real + 1j*fuzzy_imag

def row_col_from_condensed_index(d,index):
    # Credits to: https://stackoverflow.com/a/14839010
    b = 1 - (2 * d) 
    i = (-b - math.sqrt(b ** 2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return (int(i),int(j))

def parse_cplx_eig(vec: np.ndarray):
    _real_eigs_mask = (np.abs(vec.imag) <= np.finfo(vec.dtype).eps)
    real_eigs = vec[_real_eigs_mask].real
    _cplx_eigs_mask = np.logical_not(_real_eigs_mask)
    cplx_eigs = vec[_cplx_eigs_mask]
    cplx_conj_pairs_idxs = _parse_cplx_conj_pairs(cplx_eigs)
    return np.concatenate([np.sort(real_eigs), np.sort(cplx_eigs[cplx_conj_pairs_idxs])])

def _parse_cplx_conj_pairs(cplx_conj_vec: np.ndarray):
    if not cplx_conj_vec.shape[0] % 2 == 0:
        raise ValueError(
            f"The array must consist in a set of complex conjugate pairs, but its shape ({cplx_conj_vec.shape[0]}"
            f" is odd).")
    _v_sort = np.argsort(cplx_conj_vec)
    _v_cj_sort = np.argsort(cplx_conj_vec.conj())

    _diff = cplx_conj_vec[_v_sort] - cplx_conj_vec.conj()[_v_cj_sort]
    if not np.allclose(_diff, np.zeros_like(_diff)):
        raise ValueError("The provided array does not consists of complex conjugate pairs")

    _v = cplx_conj_vec[_v_sort]

    idx_list = []
    for i in range(cplx_conj_vec.shape[0]):
        _idx_tuple = (_v_sort[i], _v_cj_sort[i])
        _idx_tuple_r = (_idx_tuple[1], _idx_tuple[0])
        if _idx_tuple in idx_list:
            continue
        elif _idx_tuple_r in idx_list:
            continue
        else:
            if np.angle(_v[i]) >= 0:
                idx_list.append(_idx_tuple)
            else:
                idx_list.append(_idx_tuple_r)
    _pos_phase_idxs = [i[0] for i in idx_list]
    return np.asarray(_pos_phase_idxs, dtype=int)

class NotFittedError(Exception):
    pass

def check_is_fitted(obj: object, attr_list: list[str]):
    for attr in attr_list:
        if not hasattr(obj, attr):
            raise NotFittedError(f"{obj.__class__.__name__} is not fitted. Please call the 'fit' method first.")

def create_base_dir(path: os.PathLike):
    path = Path(path)
    base_path = path.parent
    if not base_path.exists():
        base_path.mkdir(parents=True)

def enforce_2d_output(fn: Callable) -> Callable:
    def _wrap(*a, **kw):
        res = fn(*a, **kw)
        res = np.asanyarray(res)
        if res.ndim <= 1:
            return np.atleast_2d(res)
        elif res.ndim == 2:
            return res
        else:
            logger.warn("The output has more than two dimensions. Flattening the trailing ones.")
            return np.reshape(res, (res.shape[0], -1))
    return _wrap

def enforce_2d_inputs(fn: Callable) -> Callable:
    def _wrap(*a):
        _reshaped_a = []
        for _a in a:
            if _a is None:
                _reshaped_a.append(None)
            else:
                _a = np.asanyarray(_a)
                if _a.ndim <= 1:
                    _reshaped_a.append(np.atleast_2d(_a))
                elif _a.ndim == 2:
                    _reshaped_a.append(_a)
                else:
                    logger.warn("The input has more than two dimensions. Flattening the trailing ones.")
                    _reshaped_a.append(np.reshape(_a, (_a.shape[0], -1)))
        res = fn(*_reshaped_a)
        return res
    return _wrap

def _parse_DMD_observables(observables, data, data_fit, lookback_len):
    #Shape checks:
    data = check_contexts(data, lookback_len, warn_len0_lookforward=True)
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