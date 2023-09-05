from typing import Callable, Iterable
from kooplearn.abc import FeatureMap
import numpy as np

class IdentityFeatureMap(FeatureMap):
    """Identity feature map returning the input as is after having standardized its shape to ``(n_samples, n_features)``.
    """
    def __call__(self, X: np.ndarray):
        # Standardize shape
        X = np.atleast_2d(X)

        return X.reshape(X.shape[0], -1)

class ConcatenateFeatureMaps(FeatureMap):
    """Concatenate multiple functions into a feature map. The functions should return Numpy arrays which can be concatenated along their last axis. Before returning, the feature maps are standardized to ``(n_samples, n_features)``.

    Args:
        feature_maps (Iterable of callables): A list of callables which return numpy arrays.
    """
    def __init__(self, feature_maps: Iterable[Callable]):
        super().__init__()
        self.feature_maps = feature_maps
    
    def __call__(self, X: np.ndarray):
        X = np.concatenate([np.atleast_2d(fm(X)) for fm in self.feature_maps], axis = -1)
        return X.reshape(X.shape[0], -1)