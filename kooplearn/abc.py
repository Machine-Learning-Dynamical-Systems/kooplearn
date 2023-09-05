from __future__ import annotations
import abc
import os
from typing import Optional, Union, Callable

import numpy as np


# Abstract base classes defining the interface to implement when extending kooplearn
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: np.ndarray, lookback_len: Optional[int] = None):
        """Fit the model to the data.

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.
            lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None, corresponding to ``lookback_len = context_len - 1``. The lookback length should be saved as a private attribute of the model, to be used in :func:`predict`, :func:`eig` and :func:`modes`.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None):
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial condition ``X``.
        
        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Array of context windows. The lookback slice defines the initial conditions from which we wish to predict, shape ``(n_init_conditions, context_len, *features_shape)``.
            t (int): Number of steps to predict (return the last one).
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_samples, context_len, *obs_features_shape)`` or ``None``. If array, it must be the observable evaluated at the training data. If ``None`` returns the predictions for the state.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, n_obs_features)``.
        """
        pass

    @abc.abstractmethod
    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (numpy.ndarray or None): Array of context windows used to to evaluate the left eigenfunctions on, shape ``(n_samples, context_len, *features_shape)``.
            eval_right_on (numpy.ndarray or None): Array of context windows used to evaluate the right eigenfunctions on, shape ``(n_samples, context_len, *features_shape)``.

        Returns:
            numpy.ndarray or tuple: (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. Left eigenfunctions evaluated at ``eval_left_on``, shape ``(n_samples, rank)`` if ``eval_left_on`` is not ``None``. Right eigenfunction evaluated at ``eval_right_on``, shape ``(n_samples, rank)`` if ``eval_right_on`` is not ``None``.
        """
        pass

    @abc.abstractmethod
    def save(self, path: os.PathLike):
        pass
    
    @classmethod
    @abc.abstractmethod
    def load(path: os.PathLike):
        pass

    @abc.abstractmethod
    def modes(self, data: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        """
        Computes the mode decomposition of the Koopman/Transfer operator of one or more observables of the system at the state ``X``.

        Args:
            data (numpy.ndarray): Array of context windows. The lookback slice defines the initial conditions out of which the modes are computed, shape ``(n_init_conditions, context_len, *features_shape)``..
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_samples, context_len, *obs_features_shape)`` or ``None``. If array, it must be the observable evaluated at the training data. If ``None`` returns the predictions for the state.

        Returns:
            numpy.ndarray: Modes of the system at the state ``X``, shape ``(self.rank, n_states, ...)``.
        """
        pass

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model is fitted.

        Returns:
            bool: Returns ``True`` if the model is fitted, ``False`` otherwise.
        """
        pass

class FeatureMap(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass

class TrainableFeatureMap(FeatureMap):
    @abc.abstractmethod
    def fit(self, *a, lookback_len: Optional[int] = None, **kw) -> None:
        """Fit the feature map to the data.
        """
        pass
    
    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    def initialize(self):
        pass
