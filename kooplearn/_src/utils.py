from typing import NamedTuple
import numpy as np
import os
from pathlib import Path

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

def parse_cplx_eig(vec: np.ndarray):
    _real_eigs_mask = (vec.imag == 0.)
    real_eigs = vec[_real_eigs_mask]
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