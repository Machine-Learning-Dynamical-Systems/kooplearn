from math import sqrt
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist


def topk(vec: np.ndarray, k: int):
    """Get the top k values from a Numpy array.

    Args:
        vec (ndarray): A 1D numpy array
        k (int): Number of elements to keep

    Returns:
        values, indices: top k values and their indices
    """
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = np.flip(np.argsort(vec))  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return values, indices


def directed_hausdorff_distance(pred: np.ndarray, reference: np.ndarray):
    """One-sided hausdorff distance between sets."""
    pred = np.asanyarray(pred)
    reference = np.asanyarray(reference)
    assert pred.ndim == 1
    assert reference.ndim == 1

    distances = np.zeros((pred.shape[0], reference.shape[0]), dtype=np.float64)
    for pred_idx, pred_pt in enumerate(pred):
        for reference_idx, reference_pt in enumerate(reference):
            distances[pred_idx, reference_idx] = np.abs(pred_pt - reference_pt)
    hausdorff_dist = np.max(np.min(distances, axis=1))
    return hausdorff_dist


def find_complex_conjugates(
    complex_vec: np.ndarray[np.complexfloating], tol: float = 10.0
) -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
    """
    Identify complex conjugate pairs and real eigenvalues in an array.

    This function finds pairs of complex conjugate eigenvalues by comparing
    their real parts using a tolerance based on machine epsilon. Elements
    with negligible imaginary parts are classified as real.

    Parameters
    ----------
    complex_vec : np.ndarray of complex
        1D array of complex eigenvalues to analyze
    tol : float, optional
        Tolerance multiplier for machine epsilon. Default is 10.
        The actual tolerance is ``tol * eps`` where eps is the machine
        epsilon for the array's dtype.

    Returns
    -------
    cc_pairs : np.ndarray of int, shape (n_pairs, 2)
        Array containing indices of complex conjugate pairs. Each row [i, j]
        indicates that ``complex_vec[i]`` and ``complex_vec[j]`` form a
        conjugate pair.
    real_idxs : np.ndarray of int, shape (n_real,)
        Array of indices for eigenvalues that are real (have negligible
        imaginary parts).

    Raises
    ------
    AssertionError
        If input array does not have a complex dtype
    ValueError
        If input array is not 1-dimensional, or if complex elements exist
        without matching conjugate pairs

    Notes
    -----
    The algorithm works as follows:

    1. Extract real parts and compute pairwise distances using scipy's pdist
    2. Identify pairs with real part distance < ``tol * eps``
    3. Classify remaining elements as real if imaginary part ≈ 0
    4. Raise error if unpaired complex elements exist

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> values = np.array([1+2j, 1-2j, 3.0+0j, 4+1j, 4-1j])
        >>> cc_pairs, real_idxs = find_complex_conjugates(values)
        >>> print(cc_pairs)
        [[0 1]
         [3 4]]
        >>> print(real_idxs)
        [2]
    """
    # Validate input type
    assert issubclass(complex_vec.dtype.type, np.complexfloating), (
        "The input element should be complex"
    )

    # Validate input dimensionality
    if complex_vec.ndim != 1:
        raise ValueError("The input vector should have dimension 1")

    # Extract dimension and compute relative tolerance
    d = complex_vec.shape[0]
    rcond = tol * np.finfo(complex_vec.dtype).eps

    # Compute pairwise distances between real parts
    pdist_real_part = pdist(complex_vec.real[:, None])

    # Find pairs with matching real parts (distance < tolerance)
    condensed_idxs = np.argwhere(pdist_real_part < rcond)[:, 0]

    # Convert condensed indices to row-column pairs
    cc_pairs = np.array([row_col_from_condensed_index(d, i) for i in condensed_idxs])

    # Identify indices not part of any conjugate pair
    maybe_real_idxs = np.setdiff1d(np.arange(d, dtype=int), cc_pairs.flatten())

    # Verify that remaining elements are actually real
    if not np.allclose(np.imag(complex_vec[maybe_real_idxs]), 0):
        raise ValueError(
            "The input vector contains some complex element with no matching "
            "complex conjugate pair."
        )

    return cc_pairs, maybe_real_idxs


def fuzzy_parse_complex(vec: np.ndarray, tol: float = 10.0):
    assert issubclass(vec.dtype.type, np.complexfloating), (
        "The input element should be complex"
    )
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
    i = (-b - sqrt(b**2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return (int(i), int(j))


def spd_neg_pow(
    M: np.ndarray,
    exponent: float = -1.0,
    cutoff: Optional[float] = None,
    strategy: str = "trunc",
) -> np.ndarray:
    """
    Truncated eigenvalue decomposition of A
    """
    if cutoff is None:
        cutoff = 10.0 * M.shape[0] * np.finfo(M.dtype).eps
    w, v = np.linalg.eigh(M)
    if strategy == "trunc":
        sanitized_w = np.where(w <= cutoff, 1.0, w)
        inv_w = np.where(
            w > cutoff, (sanitized_w ** np.abs(exponent)) ** np.sign(exponent), 0.0
        )
        v = np.where(w > cutoff, v, 0.0)
    elif strategy == "tikhonov":
        inv_w = ((w + cutoff) ** np.abs(exponent)) ** np.sign(exponent)
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")
    return np.linalg.multi_dot([v, np.diag(inv_w), v.T])


def covariance(X: np.ndarray, Y: Optional[np.ndarray] = None):
    X = np.atleast_2d(X)
    if X.ndim > 2:
        raise ValueError(f"Input array has more than 2 dimensions ({X.ndim}).")
    rnorm = (X.shape[0]) ** (-0.5)
    X = X * rnorm

    if Y is None:
        c = X.T @ X
    else:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Shape mismatch: the covariance between two arrays can be computed only if they have the same initial dimension. Got {X.shape[0]} and {Y.shape[0]}."
            )
        Y = np.atleast_2d(Y)
        if Y.ndim > 2:
            raise ValueError(f"Input array has more than 2 dimensions ({Y.ndim}).")
        Y = Y * rnorm
        c = X.T @ Y
    return c


def check_torch_deps():
    try:
        import torch
    except ImportError:
        raise ImportError(
            "To use kooplearn's deep learning losses please reinstall it with the `torch` extra flag by typing `pip install kooplearn[torch]`."
        )


def check_jax_deps():
    try:
        import jax
    except ImportError:
        raise ImportError(
            "To use kooplearn's deep learning losses please reinstall it with the `jax` extra flag by typing `pip install kooplearn[jax]`."
        )
