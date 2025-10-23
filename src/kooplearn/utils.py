from math import sqrt
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


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


# def covariance(
#     X: np.ndarray,
#     Y: np.ndarray | None = None,
#     center: bool = True,
#     norm: float | None = None,
# ) -> np.ndarray:
#     """
#     Compute the covariance of ``X`` or the cross-covariance between ``X`` and ``Y``.

#     Parameters
#     ----------
#     X : ndarray of shape (n_samples, n_features)
#         Input features.
#     Y : ndarray of shape (n_samples, n_features), optional
#         Output features. If provided, computes the cross-covariance between ``X`` and ``Y``.
#         If ``None``, computes the auto-covariance of ``X``.
#     center : bool, default=True
#         Whether to subtract the mean before computing the covariance.
#     norm : float, optional
#         Normalization factor. If ``None``, normalizes by ``sqrt(n_samples)``.

#     Returns
#     -------
#     ndarray of shape (n_features, n_features)
#         Covariance or cross-covariance matrix.

#     Notes
#     -----
#     Given samples :math:`X \\in \\mathbb{R}^{N \\times D}`, the (centered) covariance is:

#     .. math::

#         C_X = \\frac{1}{N} (X - \\bar{X})^T (X - \\bar{X})

#     Similarly, if ``Y`` is provided:

#     .. math::

#         C_{XY} = \\frac{1}{N} (X - \\bar{X})^T (Y - \\bar{Y})

#     """
#     assert X.ndim == 2, "X must be a 2D array."
#     n_samples = X.shape[0]

#     if norm is None:
#         norm = sqrt(n_samples)
#     else:
#         assert norm > 0, "norm must be positive."
#         norm = sqrt(norm)

#     X = X / norm

#     if Y is None:
#         if center:
#             X = X - X.mean(axis=0, keepdims=True)
#         return X.T @ X
#     else:
#         assert Y.ndim == 2, "Y must be a 2D array."
#         Y = Y / norm
#         if center:
#             X = X - X.mean(axis=0, keepdims=True)
#             Y = Y - Y.mean(axis=0, keepdims=True)
#         return X.T @ Y


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


class TimeDelayEmbedding(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that constructs time-delay embeddings
    (temporal windows) from trajectories, with configurable stride.

    Each output sample corresponds to a flattened temporal window of length
    :math:`H`, with a stride :math:`s` between the starting points of consecutive windows.

    Parameters
    ----------
    history_length : int
        Number of consecutive time steps per embedding window (:math:`H`).
    stride : int, default=1
        Step between the starts of successive windows (:math:`s`).

    Attributes
    ----------
    n_samples_in_ : int
        Number of samples in the input data seen during fitting.
    n_features_in_ : int
        Number of features per sample in the input data.

    Notes
    -----
    - The `inverse_transform` method **only works when `stride=1`**.
      Using `stride>1` will raise a ValueError, because reconstruction requires overlapping windows.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.arange(20).reshape(10, 2)
    >>> tde = TimeDelayEmbedding(history_length=3, stride=1)
    >>> X = tde.fit_transform(traj)
    >>> X.shape
    (8, 6)
    >>> reconstructed = tde.inverse_transform(X)
    >>> np.allclose(traj, reconstructed, atol=1e-8)
    True

    >>> tde2 = TimeDelayEmbedding(history_length=3, stride=2)
    >>> X2 = tde2.fit_transform(traj)
    >>> X2.shape
    (4, 6)
    >>> # inverse_transform not supported for stride > 1
    >>> tde2.inverse_transform(X2)
    ValueError: inverse_transform only works when stride=1.
    """

    def __init__(self, history_length: int, stride: int = 1):
        self.history_length = history_length
        self.stride = stride

    def fit(self, X, y=None):
        """Store input shape."""
        X = check_array(X, ensure_2d=True, dtype=float)
        self.n_samples_in_, self.n_features_in_ = X.shape
        return self

    def transform(self, X):
        """Construct time-delay embedding with stride."""
        check_is_fitted(self, ["n_samples_in_", "n_features_in_"])
        X = check_array(X, ensure_2d=True, dtype=float)

        n_samples = X.shape[0]
        if self.history_length > n_samples:
            raise ValueError("history_length must not exceed number of samples.")
        if self.stride < 1:
            raise ValueError("stride must be a positive integer.")

        n_windows = (n_samples - self.history_length) // self.stride + 1
        indices = (
            np.arange(self.history_length)[None, :]
            + self.stride * np.arange(n_windows)[:, None]
        )
        windows = X[indices]  # shape: (n_windows, history_length, n_features)
        X_embedded = windows.reshape(n_windows, -1)
        return X_embedded

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """Reconstruct approximate trajectory from flattened embeddings.

        Only works if stride=1.
        """
        check_is_fitted(self, ["n_samples_in_", "n_features_in_"])
        X = np.asarray(X, dtype=float)

        if self.stride != 1:
            raise ValueError("inverse_transform only works when stride=1.")

        expected_width = self.history_length * self.n_features_in_
        if X.ndim != 2 or X.shape[1] != expected_width:
            raise ValueError(
                f"Input must have shape (n_windows, {expected_width}), got {X.shape} instead."
            )

        n_windows = X.shape[0]
        reconstructed_length = (n_windows - 1) * self.stride + self.history_length
        reconstructed = np.zeros((reconstructed_length, self.n_features_in_))
        counts = np.zeros(reconstructed_length)

        for i in range(n_windows):
            start = i * self.stride
            end = start + self.history_length
            reconstructed[start:end] += X[i].reshape(
                self.history_length, self.n_features_in_
            )
            counts[start:end] += 1

        reconstructed /= np.maximum(counts[:, np.newaxis], 1)
        return reconstructed


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


class Flatten3DTo2DTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save shape for inverse_transform
        self._original_shape = X.shape
        return self

    def transform(self, X, y=None):
        # Flatten the last two dimensions
        n_samples = X.shape[0]
        self._rest_shape = X.shape[1:]
        return X.reshape(n_samples, -1)

    def inverse_transform(self, X, y=None):
        # Restore to original 3D shape
        n_samples = X.shape[0]
        return X.reshape((n_samples,) + self._rest_shape)


class Flattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save shape for inverse_transform
        self._feature_shape = X.shape[1:]
        return self

    def transform(self, X, y=None):
        # Flatten the last two dimensions
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def inverse_transform(self, X, y=None):
        # Restore to original 3D shape
        n_samples = X.shape[0]
        return X.reshape((n_samples,) + self._feature_shape)
