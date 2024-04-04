import logging
import math
from typing import NamedTuple

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger("kooplearn")


# Exceptions
class NotFittedError(Exception):
    pass


class ShapeError(Exception):
    pass


# Misc Utils
def check_is_fitted(obj: object, attr_list: list[str]):
    for attr in attr_list:
        if not hasattr(obj, attr):
            raise NotFittedError(
                f"Attribute \"{attr}\" not found. {obj.__class__.__name__} is not fitted. Please call the 'fit' "
                f"method first."
                )


# Sorting and parsing
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
    assert issubclass(
        vec.dtype.type, np.complexfloating
        ), "The input element should be complex"
    rcond = tol * np.finfo(vec.dtype).eps
    pdist_real_part = pdist(vec.real[:, None])
    # Set the same element whenever pdist is smaller than eps*tol
    condensed_idxs = np.argwhere(pdist_real_part < rcond)[:, 0]
    fuzzy_real = vec.real.copy()
    if condensed_idxs.shape[0] >= 1:
        for idx in condensed_idxs:
            i, j = row_col_from_condensed_index(vec.real.shape[0], idx)
            avg = 0.5 * (fuzzy_real[i] + fuzzy_real[j])
            fuzzy_real[i] = avg
            fuzzy_real[j] = avg
    fuzzy_imag = vec.imag.copy()
    fuzzy_imag[np.abs(fuzzy_imag) < rcond] = 0.0
    return fuzzy_real + 1j * fuzzy_imag


def row_col_from_condensed_index(d, index):
    # Credits to: https://stackoverflow.com/a/14839010
    b = 1 - (2 * d)
    i = (-b - math.sqrt(b ** 2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return (int(i), int(j))


def parse_cplx_eig(vec: np.ndarray):
    """ Parse and sort complex and real eigenvalues from an array.

    This function separates real and complex eigenvalues from the input array, sorts them, and returns the sorted
    eigenvalues along with their original indices in the input array.

    Parameters
    ----------
    vec : np.ndarray
        An array of complex numbers that includes real and complex eigenvalues.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the sorted real eigenvalues, the sorted complex eigenvalues, the original indices of the
        sorted real eigenvalues, and the original indices of the sorted complex eigenvalues.

    Example
    --------
    >>> import numpy as np
    >>> vec = np.array([4, 1, 0.1 + 1j, 0.1 - 1j, 1 + 1j, 1 - 1j])
    >>> real_eigs, cplx_eigs, real_idx, cplx_idx = parse_cplx_eig(vec)
    >>> real_eigs
    array([1., 4.])
    >>> real_idx
    array([1, 0])
    >>> cplx_eigs
    array([0.1+1.j, 1. +1.j])
    >>> cplx_idx
    array([2, 4])
    """
    _real_eigs_mask = np.abs(vec.imag) <= np.finfo(vec.dtype).eps
    real_eigs = vec[_real_eigs_mask].real
    real_eigs_indices = np.where(_real_eigs_mask)[0]  # track original indices for real eigenvalues

    _cplx_eigs_mask = np.logical_not(_real_eigs_mask)
    cplx_eigs = vec[_cplx_eigs_mask]
    cplx_eigs_indices = np.where(_cplx_eigs_mask)[0]  # track original indices for complex eigenvalues

    cplx_conj_pairs_idxs = idx_of_positive_phase_complex_pairs(cplx_eigs)

    # Get the indices that would sort the real and complex eigenvalues
    real_eigs_sort_idxs = np.argsort(real_eigs)
    cplx_eigs_sort_idxs = np.argsort(cplx_eigs[cplx_conj_pairs_idxs])

    # Use the indices to get the sorted eigenvalues
    real_eigs = real_eigs[real_eigs_sort_idxs]
    cplx_eigs = cplx_eigs[cplx_conj_pairs_idxs][cplx_eigs_sort_idxs]

    # Get the original indices of the sorted eigenvalues
    real_eigs_indices = real_eigs_indices[real_eigs_sort_idxs]
    cplx_eigs_indices = cplx_eigs_indices[cplx_conj_pairs_idxs][cplx_eigs_sort_idxs]

    # Get the original indices of the sorted eigenvalues
    return real_eigs, cplx_eigs, real_eigs_indices, cplx_eigs_indices

def idx_of_positive_phase_complex_pairs(cplx_conj_vec: np.ndarray):
    """Identify the indices of the elements with positive phase in an array of complex conjugate pairs.

    This function takes an array of complex numbers that are expected to be in conjugate pairs. It checks if the array
    consists of complex conjugate pairs and if the array length is even (as it should be for pairs). It then sorts the
    array and returns the indices of the elements with positive phase.

    Parameters
    ----------
    cplx_conj_vec : np.ndarray
        An array of complex numbers that are expected to be in conjugate pairs.

    Returns
    -------
    np.ndarray
        An array of indices of the elements with positive phase.

    Raises
    ------
    ValueError
        If the array does not consist of complex conjugate pairs or if its length is odd.

    Example
    --------
    >>> import numpy as np
    >>> cplx_conj_vec = np.array([1+2j, 1-2j, 2+3j, 5+0.1j, 2-3j, 5-0.1j])
    >>> indices = idx_of_positive_phase_complex_pairs(cplx_conj_vec)
    >>> indices
    array([0, 2, 3])
    >>> cplx_conj_vec[indices]
    array([1.+2.j , 2.+3.j , 5.+0.1j])
    """
    if not cplx_conj_vec.shape[0] % 2 == 0:
        raise ValueError(
            f"The array must consist in a set of complex conjugate pairs, but its shape ({cplx_conj_vec.shape[0]}"
            f" is odd)."
            )
    _v_sort = np.argsort(cplx_conj_vec)
    _v_cj_sort = np.argsort(cplx_conj_vec.conj())

    _diff = cplx_conj_vec[_v_sort] - cplx_conj_vec.conj()[_v_cj_sort]
    if not np.allclose(_diff, np.zeros_like(_diff)):
        raise ValueError(
            "The provided array does not consists of complex conjugate pairs"
            )

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
