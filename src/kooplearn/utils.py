import numpy as np


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



# def _contexts_from_traj_np(trajectory, context_length: int, time_lag: int):
#     window_shape = 1 + (context_length - 1) * time_lag
#     if window_shape > trajectory.shape[0]:
#         raise ValueError(
#             f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of "
#             f"length {trajectory.shape[0]}. Try reducing context_length or time_lag."
#         )

#     data = np.lib.stride_tricks.sliding_window_view(trajectory, window_shape, axis=0)

#     idx_map = np.lib.stride_tricks.sliding_window_view(
#         np.arange(trajectory.shape[0], dtype=np.int_).reshape(-1, 1),
#         window_shape,
#         axis=0,
#     )

#     idx_map = np.moveaxis(idx_map, -1, 1)[:, ::time_lag, ...]
#     data = np.moveaxis(data, -1, 1)[:, ::time_lag, ...]
#     return data, idx_map

# def _split_traj(trajectory, context_length: int, time_lag: int):
#     """Splits a trajectory into contexts and targets.

#     Args:
#         trajectory (ndarray): A 2D numpy array of shape (n_samples, n_features).
#         context_length (int): The length of the context.
#         time_lag (int): The time lag between consecutive elements in the context.

#     Returns:
#         X (ndarray): A 2D numpy array of shape (n_samples - (context_length - 1) * time_lag, context_length * n_features) containing the contexts.
#         Y (ndarray): A 2D numpy array of shape (n_samples - (context_length - 1) * time_lag, n_features) containing the targets.
#     """
#     if trajectory.ndim != 2:
#         raise ValueError("trajectory must be a 2D array")
#     if context_length < 1:
#         raise ValueError("context_length must be at least 2")
#     if time_lag < 1:
#         raise ValueError("time_lag must be at least 1")

#     data, _ = _contexts_from_traj_np(trajectory, context_length, time_lag)
#     n_samples, _, n_features = data.shape
#     X = data[:, :-1, :].reshape(n_samples, (context_length - 1) * n_features)
#     Y = data[:, -1, :].reshape(n_samples, n_features)
#     return X, Y