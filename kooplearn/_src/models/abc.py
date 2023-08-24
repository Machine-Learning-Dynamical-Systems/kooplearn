from __future__ import annotations
import abc
import os
from typing import Optional, Union, Callable

import numpy as np
from numpy.typing import ArrayLike

# Abstract base classes defining the interface to implement when extending kooplearn
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: ArrayLike, Y: ArrayLike):
        pass

    @abc.abstractmethod
    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        pass

    @abc.abstractmethod
    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        pass

    @abc.abstractmethod
    def save(self, path: os.PathLike):
        pass
    
    @classmethod
    @abc.abstractmethod
    def load(path: os.PathLike):
        pass

    @abc.abstractmethod
    def modes(self, Xin: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        pass

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass



class FeatureMap(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass

    def cov(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        phi_X = self.__call__(X)
        if Y is None:
            c = phi_X.T @ phi_X
        else:
            phi_Y = self.__call__(Y)
            c = phi_X.T @ phi_Y
        c *= (X.shape[0]) ** (-1)
        return c


class IdentityFeatureMap(FeatureMap):
    def __call__(self, X: np.ndarray):
        return X


class TrainableFeatureMap(FeatureMap):
    # Trainable feature maps should be callable with numpy arrays and return numpy arrays (see FeatureMap abc).
    # Internally they can do whatever.
    @abc.abstractmethod
    def fit(self, *a, **kw) -> None:
        pass
    
    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    def initialize(self):
        pass
