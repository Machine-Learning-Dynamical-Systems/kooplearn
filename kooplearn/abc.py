from __future__ import annotations

import abc
import os
from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

from kooplearn._src.serialization import pickle_load, pickle_save


# Abstract base classes defining the interface to implement when extending kooplearn
class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, *a, **kw):
        """Fit the model to the data. The signature of this function must be specified from the derived class. For example, in :class:`kooplearn.models.Nonlinear`, the signature is ``fit(self, data: np.ndarray)``, while in -TODO add AutoEncoder example."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        data: ContextWindowDataset,
        t: int = 1,
        predict_observables: bool = True,
        reencode_every: int = 0,
    ):
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``data.lookback(self.lookback_len)`` being the lookback slice of ``data``.
        If ``data.observables`` is not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (ContextWindowDataset): Dataset of context windows. The lookback window of ``data`` will be used as the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``data.observables``, if present. Default to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict`` will contain the special key ``__state__`` containing the prediction for the state as well.
        """
        pass

    @abc.abstractmethod
    def eig(
        self,
        eval_left_on: Optional[ContextWindowDataset] = None,
        eval_right_on: Optional[ContextWindowDataset] = None,
    ):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (ContextWindowDataset or None): Dataset of context windows on which the left eigenfunctions are evaluated.
            eval_right_on (ContextWindowDataset or None): Dataset of context windows on which the right eigenfunctions are evaluated.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on`` are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on`` / ``eval_right_on`` : shape ``(n_samples, rank)``.
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
        data: ContextWindowDataset,
        predict_observables: bool = True,
    ):
        """
        Computes the mode decomposition of arbitrary observables of the Koopman/Transfer operator at the states defined by ``data``.

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as the initial condition, see the note above.
            predict_observables (bool): Return the prediction for the observables in ``data.observables``, if present. Default to ``True``.
        Returns:
            (modes, eigenvalues): Modes and corresponding eigenvalues of the system at the states defined by ``data``. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict`` will contain the special key ``__state__`` containing the modes for the state as well.
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


class ContextWindow(Sequence):
    """
    Class for a single context window, i.e. the :ref:`kooplearn's data paradigm <kooplearn_data_paradigm>`.
    """

    def __init__(self, window: Sequence):
        """
        Initializes the context window.

        Args:
            window (Sequence): A sequence of data points.
        """
        self.data = window
        self._context_length = len(window)

    @property
    def context_length(self):
        """
        Returns the length of the context window.
        """
        return self._context_length

    def slice(self, slice_obj):
        """
        Returns a slice of the context window given a slice object.

        Args:
            slice_obj (slice): The python slice function.

        Returns:
            Slice of the context window.
        """
        return self.data[slice_obj]

    def __len__(self):
        """
        Returns the length of the context window.
        """
        return 1  # Single context window

    def lookback(self, lookback_length: int, slide_by: int = 0):
        """
        Returns the lookback window of the context window.

        Args:
            lookback_length (int):  Length of the lookback window.
            slide_by (int, optional): Number of slides along the context window. Default to ``0``.

        Returns:
            Lookback window of the context window.
        """
        self._check_lb_len(lookback_length)
        max_slide = self._context_length - lookback_length
        if slide_by > max_slide:
            raise ValueError(
                f"Invalid slide_by = {slide_by} for lookback_length = {lookback_length} and Context of length = {self._context_length}. It should be 0 <= slide_by <= context_length - lookback_length"
            )

        lb_window = self.slice(slice(slide_by, lookback_length + slide_by))
        return lb_window

    def lookforward(self, lookback_length: int):
        """
        Returns the lookforward window of the context window.

        Args:
            lookback_length (int): Length of the lookback window.

        Returns:
            Lookforward window of the context window.
        """
        self._check_lb_len(lookback_length)
        lf_window = self.slice(slice(lookback_length, None))
        return lf_window

    def _check_lb_len(self, lookback_length: int):
        """
        Checks the validity of the lookback length.

        Args:
            lookback_length (int): Length of the lookback window.
        """
        if (lookback_length > self.context_length) or (lookback_length < 1):
            raise ValueError(
                f"Invalid lookback_length = {lookback_length} for ContextWindow of length = {self.context_length}. It should be 1 <= lookback_length <= context_length."
            )

    def __repr__(self):
        """
        Returns a string representation of the context window.
        """
        return f"ContextWindow <context_length={self.context_length}, data={self.data.__str__()}>"

    def __getitem__(self, idx):
        """
        Returns the context window's point at the given index.

        Args:
            idx (int): The index of the point to be returned.

        Returns:
            The point at the given index.
        """
        return self.data[idx]

    def save(self, path: os.PathLike):
        """
        Saves the current context window to the given path.

        Args:
            path (path-like): Save the context window to path.
        """
        pickle_save(self, path)

    @classmethod
    def load(cls, filename):
        """
        Loads the context window from the given filename.

        Args:
            filename: Load the context window from the given filename.

        Returns:
            The context window object.
        """
        return pickle_load(cls, filename)


class ContextWindowDataset(ContextWindow):
    """
    Class for a collection of :obj:`ContextWindow` objects.
    """

    def __init__(self, dataset: Iterable[Sequence]):
        """
        Initializes the context window dataset.

        Args:
            dataset (Iterable[Sequence]): A sequence of :obj:`ContextWindow` objects.
        """
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
        """
        Returns the number of context windows in the dataset.
        """
        return len(self.data)

    def __iter__(self):
        """
        Returns an iterator over the context windows in the dataset
        """
        return iter([ContextWindow(ctx) for ctx in self.data])

    def __getitem__(self, idx):
        """
        Returns the context window at the given index.

        Args:
            idx: The index of the context window to be returned.

        Returns:
            The context window at the given index.
        """
        return ContextWindow(self.data[idx])

    def __repr__(self):
        """
        Returns a string representation of the context window dataset.
        """
        return f"{self.__class__.__name__} <item_count={len(self)}, context_length={self.context_length}, data={self.data.__str__()}>"

    def slice(self, slice_obj):
        """
        Returns a slice of the context window dataset given a slice object.

        Args:
            slice_obj (slice): The python slice object.

        Returns:
            Slice of the context window dataset.
        """
        return [ctx[slice_obj] for ctx in self.data]
