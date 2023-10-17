from __future__ import annotations

import logging
from typing import Optional

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import NotFittedError
from kooplearn.abc import TrainableFeatureMap
from kooplearn.models import ExtendedDMD

logger = logging.getLogger("kooplearn")


class DeepEDMD(ExtendedDMD):
    """
    Deep Extended Dynamic Mode Decomposition (DeepEDMD) Model.

    Implements the Extended Dynamic Mode Decomposition estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`. In Deep Extended Dynamic Mode Decomposition the feature map used to embed the data is learned as well. :guilabel:`TODO - Add different refs`

    This model implements every method of :class:`kooplearn.models.ExtendedDMD`.

    .. caution::

        The feature map passed as a first argument should be already trained, that is ``feature_map.is_fitted == True``. If this is not the case, a ``NotFittedError`` is raised.

    Args:
        feature_map (callable): *Trained* feature map used for the DeepEDMD algorithm. Should be a subclass of :class:`kooplearn.abc.TrainableFeatureMap`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. ``None`` returns the full rank estimator.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.
    """

    def __init__(
        self,
        feature_map: TrainableFeatureMap,
        reduced_rank: bool = True,
        rank: Optional[int] = None,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        rng_seed: Optional[int] = None,
    ):
        # Check that the provided feature map is trainable
        assert hasattr(
            feature_map, "fit"
        ), "The provided feature map is not trainable. Please provide a subclass of kooplearn.abc.TrainableFeatureMap."
        if not feature_map.is_fitted:
            raise NotFittedError(
                """
                The provided feature map is not fitted. Please call the fit method before initializing the DeepEDMD model.
                """
            )
        super().__init__(
            feature_map=feature_map,
            reduced_rank=reduced_rank,
            rank=rank,
            tikhonov_reg=tikhonov_reg,
            svd_solver=svd_solver,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            rng_seed=rng_seed,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            DeepEDMD: The loaded model.
        """
        return pickle_load(cls, filename)
