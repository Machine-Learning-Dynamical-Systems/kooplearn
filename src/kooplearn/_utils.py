from warnings import warn

import numpy as np
from scipy.spatial.distance import pdist


def stable_topk(
    vec: np.ndarray,
    k_max: int,
    rcond: float | None = None,
    ignore_warnings: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes up to k_max indices of the top k_max values of vec.

    If the values are below rcond, they are discarded.

    Parameters
    ----------
    vec : np.ndarray
        Vector to extract the top k indices from.
    k_max : int
        Number of indices to extract.
    rcond : float or None, optional
        Value below which the values are discarded. Default is None, in which case
        it is set according to the machine precision of vec's dtype.
    ignore_warnings : bool, optional
        If False, raise a warning when some elements are discarded for being below
        the requested numerical precision. Default is True.

    Returns
    -------
    values : np.ndarray
        Top k values (up to k_max, filtered by rcond).
    indices : np.ndarray
        Indices of the top k values.
    """

    def _topk(vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the top k values from a Numpy array.

        Parameters
        ----------
        vec : np.ndarray
            A 1D numpy array.
        k : int
            Number of elements to keep.

        Returns
        -------
        values : np.ndarray
            Top k values.
        indices : np.ndarray
            Indices of the top k values.
        """
        assert np.ndim(vec) == 1, "'vec' must be a 1D array"
        assert k > 0, "k should be greater than 0"
        sort_perm = np.argsort(vec)[::-1]  # descending order
        indices = sort_perm[:k]
        values = vec[indices]
        return values, indices

    if rcond is None:
        rcond = 10.0 * vec.shape[0] * np.finfo(vec.dtype).eps

    top_vec, top_idxs = _topk(vec, k_max)

    if all(top_vec > rcond):
        return top_vec, top_idxs
    else:
        valid = top_vec > rcond
        # In the case of multiple occurrences of the maximum vec, the indices corresponding to the
        # first occurrence are returned.
        first_invalid = np.argmax(~valid)
        _first_discarded_val = np.max(np.abs(top_vec[first_invalid:]))

        if not ignore_warnings:
            warn(f"Warning: Discarded {k_max - np.sum(valid)} dimensions of the {k_max} " \
                 f"requested due to numerical instability. Consider decreasing k_max. " \
                 f"The largest discarded value is: {_first_discarded_val:.3e}.")
        return top_vec[valid], top_idxs[valid]


def find_complex_conjugates(
        complex_vec: np.ndarray,
        tol: float = 10.0
        ) -> tuple[np.ndarray, np.ndarray]:
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
    assert issubclass(complex_vec.dtype.type, np.complexfloating), "The input element should be"
    " complex"

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
        raise ValueError("The input vector contains some complex element with no matching complex" \
        " conjugate pair.")

    return cc_pairs, maybe_real_idxs


def fuzzy_parse_complex(vec: np.ndarray, tol: float = 10.0) -> np.ndarray:
    """Average the real parts of complex numbers that are close to each other.

    This function identifies complex numbers in a vector whose real parts are
    closer than a given tolerance. It then averages their real parts.
    Imaginary parts close to zero are set to zero.

    Parameters
    ----------
    vec : np.ndarray
        A 1D array of complex numbers.
    tol : float, optional
        Tolerance multiplier for machine epsilon. The actual tolerance is
        `tol * eps`, where `eps` is the machine epsilon for the array's
        dtype. Default is 10.0.

    Returns
    -------
    np.ndarray
        A new array with the real parts of close complex numbers averaged
        and imaginary parts close to zero set to zero.
    """
    assert issubclass(vec.dtype.type, np.complexfloating), "The input element should be complex"
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


def row_col_from_condensed_index(d: int, index: int) -> tuple[int, int]:
    """Convert a condensed index to a row-column index.

    Converts a condensed index from `scipy.spatial.distance.pdist` to a
    row-column index in a square matrix.

    Parameters
    ----------
    d : int
        The dimension of the square matrix.
    index : int
        The condensed index.

    Returns
    -------
    i : int
        Row index.
    j : int
        Column index.
    """
    # Credits to: https://stackoverflow.com/a/14839010
    b = 1 - (2 * d)
    i = (-b - np.sqrt(b**2 - 8 * index)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return int(i), int(j)


def check_torch_deps():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError("To use kooplearn's deep learning losses please reinstall it with" \
        " the `torch` extra flag by typing `pip install kooplearn[torch]`.")


def check_jax_deps():
    try:
        import jax  # noqa: F401
    except ImportError:
        raise ImportError("To use kooplearn's deep learning losses please reinstall it with" \
        " the `jax` extra flag by typing `pip install kooplearn[jax]`.")
