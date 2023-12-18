from typing import Callable, Iterable

import numpy as np

from kooplearn.abc import FeatureMap


class IdentityFeatureMap(FeatureMap):
    """Identity feature map returning the input as is."""

    def __call__(self, X: np.ndarray):
        # Standardize shape
        X = np.atleast_2d(X)
        return X


class ConcatenateFeatureMaps(FeatureMap):
    """Concatenate multiple functions into a feature map. The functions should return Numpy arrays which can be concatenated along their last axis.

    Args:
        feature_maps (Iterable of callables): A list of callables which return numpy arrays.
    """

    def __init__(self, feature_maps: Iterable[Callable]):
        super().__init__()
        self.feature_maps = feature_maps

    def __call__(self, X: np.ndarray):
        # Standardize shape
        X = np.asanyarray(X)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        X = np.concatenate([fm(X) for fm in self.feature_maps], axis=-1)
        return X
