from __future__ import annotations

import abc
from typing import Optional, Union, Callable

import numpy as np
from numpy.typing import ArrayLike
import pickle

from sklearn.utils.validation import check_is_fitted


# Abstract base classes defining the interface to implement when extending kooplearn


class BaseModel(abc.ABC):
    def __init__(self, rank, tikhonov_reg, svd_solver, iterated_power, n_oversamples, optimal_sketching):
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching
        self.Y_fit_ = None
        self.X_fit_ = None
        self._eig_cache = None
        # self.V_ = None
        # self.U_ = None

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
    def modes(self, Xin: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        pass

    def save(self, filename):
        with open(filename, 'wb+') as file:
            pickle.dump(self, file)

    def load(self, filename):
        try:
            with open(filename, 'rb+') as file:
                new_obj = pickle.load(file)
        except:  # TODO specify exception
            ('Unable to load {}'.format(filename))
        assert self._verify_adequacy(new_obj), "Incoherent or different parameters between models"
        check_is_fitted(new_obj)
        self.U_ = new_obj.U_.copy()
        self.V_ = new_obj.V_.copy()
        self.X_fit_ = new_obj.X_fit_.copy()
        self.Y_fit_ = new_obj.Y_fit_.copy()
        return new_obj

    def _verify_adequacy(self, new_obj: BaseModel):
        if self.rank != new_obj.rank:
            return False
        if self.tikhonov_reg != new_obj.tikhonov_reg:
            return False
        if self.svd_solver != new_obj.svd_solver:
            return False
        if self.iterated_power != new_obj.iterated_power:
            return False
        if self.n_oversamples != new_obj.n_oversamples:
            return False
        if self.optimal_sketching != new_obj.optimal_sketching:
            return False
        return True


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
