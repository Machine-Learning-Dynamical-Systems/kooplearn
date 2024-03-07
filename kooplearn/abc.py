from __future__ import annotations

import abc
import os
from collections.abc import Iterable, Sequence
from typing import Callable, Optional, Union

import numpy as np

from kooplearn._src.serialization import pickle_load, pickle_save


# Abstract base classes defining the interface to implement when extending kooplearn
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, *a, **kw):
        """Fit the model to the data. The signature of this function must be specified from the derived class. For example, in :class:`kooplearn.models.ExtendedDMD`, the signature is ``fit(self, data: np.ndarray)``, while in -TODO add AutoEncoder example."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        data: np.ndarray,
        t: int = 1,
        observables: Optional[Union[Callable, np.ndarray]] = None,
    ):
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``X = data[:, self.lookback_len:, ...]`` being the lookback slice of ``data``.

        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Initial conditions to predict. Array of context windows with shape ``(n_init_conditions, context_len, *features_shape)`` whose trailing dimensions match the dimensions of the data used in :func:`fit`.
            t (int): Number of steps to predict (return the last one).
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_init_conditions, context_len, *obs_features_shape)`` or ``None``. If array, it must be the observable evaluated at the training data. If ``None`` returns the predictions for the state.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, n_obs_features)``.
        """
        pass

    @abc.abstractmethod
    def eig(
        self,
        eval_left_on: Optional[np.ndarray] = None,
        eval_right_on: Optional[np.ndarray] = None,
    ):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (numpy.ndarray or None): Array of context windows on which the left eigenfunctions are evaluated, shape ``(n_samples, context_len, *features_shape)``.
            eval_right_on (numpy.ndarray or None): Array of context windows on which the right eigenfunctions are evaluated, shape ``(n_samples, context_len, *features_shape)``.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``  are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``: shape ``(n_samples, rank)``.
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
    def modes(
        self,
        data: np.ndarray,
        observables: Optional[Union[Callable, np.ndarray]] = None,
    ):
        """
        Computes the mode decomposition of arbitrary observables of the Koopman/Transfer operator at the states defined by ``data``.

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        Args:
            data (numpy.ndarray): Initial conditions to compute the modes on. See :func:`predict` for additional details.
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_samples, context_len, *obs_features_shape)`` or ``None``. If array, it must be the desired observable evaluated on the *lookforward slice* of the training data. If ``None`` returns the predictions for the state.
        Returns:
            Modes of the system at the states defined by ``data``. Array of shape ``(rank, n_samples, ...)``.
        """
        pass

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model is fitted.

        Returns:
            Returns ``True`` if the model is fitted, ``False`` otherwise.
        """
        pass

    @property
    @abc.abstractmethod
    def lookback_len(self) -> int:
        """Length of the lookback window associated to the contexts. Upon fitting, the dimension of the lookforward window will be inferred from the context window length and this attribute. Moreover, shape checks against this attribute will be performed on the data passed to :func:`fit`, :func:`predict`, :func:`eig` and :func:`modes`.

        Returns:
            Length of the lookback window associated to the contexts.
        """
        pass


class FeatureMap(abc.ABC):
    """Abstract base class for feature maps. The :func:`__call__` method must accept a batch of data points of shape ``(n_samples, *features_shape)`` and return a batch of features of shape ``(n_samples, out_features)``.

    .. caution::

        As described in :ref:`kooplearn's data paradigm <kooplearn_data_paradigm>`, the inputs passed to the methods of :class:`BaseModel` will be in the form of batches of context windows. :class:`FeatureMap`, instances are a notable departure from this paradigm, and we expect a feature map to be called on batches of data points, not context windows. The context windows are parsed internally by :class:`BaseModel` and passed to the feature map as batches of appropriate data points.

        This behaviour is designed to facilitate the reuse of feature maps across different models, and possibly even outside of kooplearn.
    """

    @abc.abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


class TrainableFeatureMap(FeatureMap):
    @abc.abstractmethod
    def fit(self, *a, **kw) -> None:
        """Fit the feature map to the data."""
        pass

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def lookback_len(self) -> int:
        """Length of the lookback window associated to the contexts. Upon fitting, the dimension of the lookforward window will be inferred from the context window length.

        Returns:
            Length of the lookback window associated to the contexts.
        """
        pass


class ContextWindow(Sequence):  # A single context window
    def __init__(self, window: Sequence):
        self.data = window
        self._context_length = len(window)

    @property
    def context_length(self):
        return self._context_length

    def slice(self, slice_obj):
        return self.data[slice_obj]

    def __len__(self):
        return 1  # Single context window

    def lookback(self, lookback_length: int, slide_by: int = 0):
        self._check_lb_len(lookback_length)
        max_slide = self._context_length - lookback_length
        if slide_by > max_slide:
            raise ValueError(
                f"Invalid slide_by = {slide_by} for lookback_length = {lookback_length} and Context of length = {self._context_length}. It should be 0 <= slide_by <= context_length - lookback_length"
            )

        lb_window = self.slice(slice(slide_by, lookback_length + slide_by))
        return lb_window

    def lookforward(self, lookback_length: int):
        self._check_lb_len(lookback_length)
        lf_window = self.slice(slice(lookback_length, None))
        return lf_window

    def _check_lb_len(self, lookback_length: int):
        if (lookback_length > self.context_length) or (lookback_length < 1):
            raise ValueError(
                f"Invalid lookback_length = {lookback_length} for ContextWindow of length = {self.context_length}. It should be 1 <= lookback_length <= context_length."
            )

    def __repr__(self):
        return f"ContextWindow <context_length={self.context_length}, data={self.data.__str__()}>"

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, path: os.PathLike):
        pickle_save(self, path)

    @classmethod
    def load(cls, filename):
        return pickle_load(cls, filename)


class ContextWindowDataset(ContextWindow):
    def __init__(self, dataset: Iterable[Sequence]):
        data = []
        context_lengths = []
        for ctx in dataset:
            if isinstance(ctx, Sequence):
                if isinstance(ctx, ContextWindow):
                    context_lengths.append(ctx.context_length)
                else:
                    context_lengths.append(len(ctx))
                data.append(ctx)
            else:
                raise TypeError(
                    f"{self.__class__.__name__} should be initialized with an iterable of ContextWindow objects, while an object of {type(ctx)} was provided."
                )
        context_lengths = set(context_lengths)
        if len(context_lengths) != 1:
            raise ValueError(
                "The context windows in the dataset should all have the same length."
            )
        self.data = data
        self._context_length = context_lengths.pop()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter([ContextWindow(ctx) for ctx in self.data])

    def __getitem__(self, idx):
        return ContextWindow(self.data[idx])

    def __repr__(self):
        return f"{self.__class__.__name__} <item_count={len(self)}, context_length={self.context_length}, data={self.data.__str__()}>"

    def slice(self, slice_obj):
        return [ctx[slice_obj] for ctx in self.data]
