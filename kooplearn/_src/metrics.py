import numpy as np
from scipy.spatial.distance import cdist

def directed_hausdorff_distance(X, Y, metric = 'euclidean'):
    """One-sided hausdorff distance between sets.

    Args:
        X (np.ndarray): An array of shape ``(n_points_X, dim)``
        Y (np.ndarray): An array of shape ``(n_points_Y, dim)``
        metric (str, optional): Any of the metrics accepted by ``scipy.spatial.distance.cdist``. Defaults to 'euclidean'.

    Returns:
        float: math:`\max_{x \in X}\min_{y \in Y} d(x, y)`
    """
    _distances = cdist(X, Y, metric = metric) 
    return np.max(np.min(_distances, axis=1))

