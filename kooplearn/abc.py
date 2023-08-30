from __future__ import annotations
import abc
import os
from typing import Optional, Union, Callable

import numpy as np


# Abstract base classes defining the interface to implement when extending kooplearn
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the model to the data.

        Args:
            X (np.ndarray): Fitting data consisting of a collection of snapshots.
            Y (np.ndarray): Fitting data being the one-step-ahead evolution of ``X``.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None):
        pass

    @abc.abstractmethod
    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
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
