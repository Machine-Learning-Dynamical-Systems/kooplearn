from __future__ import annotations
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union

from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y
from kooplearn._src.utils import NotFittedError
from kooplearn.abc import TrainableFeatureMap
from kooplearn.models import ExtendedDMD
import logging
logger = logging.getLogger('kooplearn')

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

    Attributes:
        X_fit : Training data of shape ``(n_samples, n_features)`` corresponding to a collection of sampled states.
        Y_fit : Evolved training data of shape ``(n_samples, n_features)`` corresponding the evolution of ``X_fit`` after one step.
        cov_X : Covariance matrix of the feature map evaluated at X_fit, shape ``(n_features, n_features)``.
        cov_Y : Covariance matrix of the feature map evaluated at Y_fit, shape ``(n_features, n_features)``.
        cov_XY : Cross-covariance matrix of the feature map evaluated at X_fit and Y_fit, shape ``(n_features, n_features)``.
        U : Projection matrix of shape (n_features, rank). The Koopman/Transfer operator is approximated as :math:`U U^T \mathrm{cov_{XY}}`.
    """

    def __init__(self, 
                feature_map: TrainableFeatureMap, 
                reduced_rank: bool = True,
                rank: Optional[int] = None, 
                tikhonov_reg: Optional[float] = None,
                svd_solver: str = 'full',
                iterated_power: int = 1,
                n_oversamples: int = 5,
                rng_seed: Optional[int] = None):
        #Check that the provided feature map is trainable
        assert hasattr(feature_map, 'fit')
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
            rng_seed=rng_seed)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None) \
            -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (numpy.ndarray or None): States of the system to evaluate the left eigenfunctions on, shape ``(n_samples, n_features)``.
            eval_right_on (numpy.ndarray or None): States of the system to evaluate the right eigenfunctions on, shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray or tuple: (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. Left eigenfunctions evaluated at ``eval_left_on``, shape ``(n_samples, rank)`` if ``eval_left_on`` is not ``None``. Right eigenfunction evaluated at ``eval_right_on``, shape ``(n_samples, rank)`` if ``eval_right_on`` is not ``None``.
        """
        if eval_left_on is not None:
            assert eval_left_on.shape[1:] == self._data_shape
            eval_left_on = np.asarray(eval_left_on).reshape(-1, np.prod(self._data_shape))
        if eval_right_on is not None:
            assert eval_right_on.shape[1:] == self._data_shape
            eval_right_on = np.asarray(eval_right_on).reshape(-1, np.prod(self._data_shape))
        
        return super().eig(eval_left_on=eval_left_on, eval_right_on=eval_right_on)

    def save(self, path: os.PathLike):
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        raise NotImplementedError

    def _pre_fit_checks(self, X: np.ndarray, Y: np.ndarray):
        """Performs pre-fit checks on the training data.

        Use check_array and check_X_y from sklearn to check the training data, initialize the covariance matrices and
        save the training data.

        Args:
            X: X training data of shape (n_samples, ...) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, ...) corresponding to the state at time t+1.

        """
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True, allow_nd=True, ensure_2d=False))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True, allow_nd=True, ensure_2d=False))
        assert X.shape[0] == Y.shape[0]

        self._data_shape = X.shape[1:]

        cov_X, cov_Y, cov_XY = self._init_covs(X, Y) #The feature map class takes care of computing the covariance (and perform shape checks if necessary)

        self.cov_X = cov_X
        self.cov_Y = cov_Y
        self.cov_XY = cov_XY

        self.X_fit = X
        self.Y_fit = Y

        if self.rank is None:
            self.rank = min(self.cov_X.shape)
            logger.info(f"Rank of the estimator set to {self.rank}")

        if hasattr(self, '_eig_cache'):
            del self._eig_cache