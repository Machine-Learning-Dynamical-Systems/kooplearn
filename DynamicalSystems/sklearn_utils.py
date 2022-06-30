import numpy as np

def sort_and_crop(vec, num_components = None):
    """Return the i

    Args:
        vec (ndarray): 1D array of floats
        num_component (int, optional): Number of indices to retain. Defaults to None corresponding to every indices.

    Returns:
        ndarray: array of integers corresponding to the indices of the largest num_components elements in vec.
    """
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    sort_perm = np.argsort(vec)[::-1] # descending order
    if num_components is None:
        return sort_perm
    else:
        return sort_perm[:num_components]