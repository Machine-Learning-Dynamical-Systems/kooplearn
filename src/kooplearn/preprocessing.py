import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class TimeDelayEmbedding(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that constructs time-delay embeddings
    (temporal windows) from trajectory data, with a configurable stride.

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
    - The ``inverse_transform`` method **only works when ``stride=1``**.
      Using ``stride>1`` will raise a ``ValueError``, because reconstruction requires overlapping
      windows.

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
        """
        Fit the transformer by storing the input data shape.

        This method validates the input array and stores its dimensions
        for later use in transformations or inverse transformations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input trajectory data.

        y : None
            Ignored. Present for API compatibility with scikit-learn pipelines.

        Returns
        -------
        self : TimeDelayEmbedding
            Fitted transformer instance.
        """
        X = check_array(X, ensure_2d=True, dtype=float)
        self.n_samples_in_, self.n_features_in_ = X.shape
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Construct the time-delay embedding of the input trajectory.

        Builds overlapping or non-overlapping temporal windows of length
        ``history_length`` with ``stride`` between successive windows.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input trajectory data to embed.

        Returns
        -------
        ndarray of shape (n_windows, history_length * n_features)
            Time-delay embedded representation of the input data.

        Raises
        ------
        ValueError
            If ``history_length`` exceeds the number of samples in ``X``.
        ValueError
            If ``stride`` is not a positive integer.
        """

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
        """
        Reconstruct input trajectory from flattened time-delay embeddings.

        This method reverses the transformation performed by
        :meth:`~TimeDelayEmbedding.transform`. It is only supported when ``stride=1``, since larger
        strides lead to non-overlapping windows and ambiguous reconstruction.

        Parameters
        ----------
        X : ndarray of shape (n_windows, history_length * n_features_in_)
            Flattened time-delay embedded data.

        Returns
        -------
        ndarray of shape (n_samples, n_features_in_)
            Approximate reconstruction of the original trajectory.

        Raises
        ------
        ValueError
            If ``stride != 1``.
        ValueError
            If input shape is incompatible with ``history_length`` and the
            number of input features.

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
    """
    A scikit-learn compatible transformer that flattens multi-dimensional trajectories
    into a 2D array, and restores them to their original shape when inverted.

    This transformer is useful when working with models that expect 2D input
    (e.g., `(n_samples, n_features)`), but the data naturally has higher-order
    structure, e.g., images or spatio-temporal fields.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from kooplearn.preprocessing import FeatureFlattener
    >>>
    >>> X = np.random.rand(10, 4, 5)  # e.g., 10 snapshots of a 4×5 field
    >>> flattener = FeatureFlattener()
    >>> X_flat = flattener.fit_transform(X)
    >>> X_flat.shape
    (10, 20)
    >>> X_reconstructed = flattener.inverse_transform(X_flat)
    >>> np.allclose(X, X_reconstructed)
    True
    """
    def fit(self, X, y=None):
        """Store the original feature shape for later reconstruction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data with arbitrary feature dimensions.

        y : None
            Ignored. Present for API compatibility with scikit-learn pipelines.

        Returns
        -------
        self : object
            Fitted transformer instance.
        """
        self._feature_shape = X.shape[1:]
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Flatten input features into a 2D array.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data to flatten.

        y : None
            Ignored. Present for API compatibility with scikit-learn pipelines.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Flattened input data.
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def inverse_transform(self, X, y=None):
        """Restore flattened features to their original shape.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Flattened data to reconstruct.

        y : None
            Ignored. Present for API compatibility with scikit-learn pipelines.

        Returns
        -------
        ndarray of shape (n_samples, ...)
            Data reshaped to the original feature dimensions.
        """
        n_samples = X.shape[0]
        return X.reshape((n_samples,) + self._feature_shape)
