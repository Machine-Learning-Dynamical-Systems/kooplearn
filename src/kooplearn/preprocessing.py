import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


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
    """

    def __init__(self, history_length: int, stride: int = 1):
        self.history_length = history_length
        self.stride = stride

    def fit(self, X, y=None):
        """Store input shape."""
        X = check_array(X, ensure_2d=True, dtype=float)
        self.n_samples_in_, self.n_features_in_ = X.shape
        self._is_fitted = True
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


class FeatureFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save shape for inverse_transform
        self._feature_shape = X.shape[1:]
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        # Flatten the last dimensions
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def inverse_transform(self, X, y=None):
        # Restore to original shape
        n_samples = X.shape[0]
        return X.reshape((n_samples,) + self._feature_shape)
