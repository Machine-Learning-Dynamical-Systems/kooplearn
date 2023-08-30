from __future__ import annotations
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Optional, Callable, Union

from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y
from kooplearn._src.utils import check_is_fitted, create_base_dir
from kooplearn.abc import BaseModel, FeatureMap
from kooplearn.models.feature_maps import IdentityFeatureMap
from kooplearn._src.operator_regression import primal
import logging
logger = logging.getLogger('kooplearn')

class ExtendedDMD(BaseModel):
    """
    Extended Dynamic Mode Decomposition (ExtendedDMD) Model.
    Implements the ExtendedDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.
    
    Parameters:
        feature_map (callable): Dictionary of functions used for the ExtendedDMD algorithm. Should be a subclass of ``kooplearn.abc.FeatureMap``.
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
                feature_map: FeatureMap = IdentityFeatureMap(), 
                reduced_rank: bool = True,
                rank: Optional[int] = None, 
                tikhonov_reg: Optional[float] = None,
                svd_solver: str = 'full',
                iterated_power: int = 1,
                n_oversamples: int = 5,
                rng_seed: Optional[int] = None):
        #Perform checks on the input arguments:
        if svd_solver not in ['full', 'arnoldi', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if svd_solver == 'randomized' and iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if svd_solver == 'randomized' and n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        self.rng_seed = rng_seed
        self.feature_map = feature_map
        self.rank = rank
        if tikhonov_reg is None:
            self.tikhonov_reg = 0.
        else:
            self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.reduced_rank = reduced_rank
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fits the ExtendedDMD model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model.

        Parameters:
            X : Training data of shape ``(n_samples, n_features)`` corresponding to a collection of sampled states.
            Y : Evolved training data of shape ``(n_samples, n_features)`` corresponding the evolution of ``X`` after one step.
        
        Returns:
            self (ExtendedDMD): The fitted estimator.
        """
        self._pre_fit_checks(X, Y)
        if self.reduced_rank:
            if self.svd_solver == 'randomized':
                vectors = primal.fit_rand_reduced_rank_regression(self.cov_X, self.cov_XY, self.tikhonov_reg, self.rank, self.n_oversamples, self.iterated_power, self.rng_seed)
            else:
                vectors = primal.fit_reduced_rank_regression(self.cov_X, self.cov_XY, self.tikhonov_reg, self.rank, self.svd_solver)
        else:
            if self.svd_solver == 'randomized':
                vectors = primal.fit_rand_principal_component_regression(self.cov_X, self.tikhonov_reg, self.rank, self.n_oversamples, self.iterated_power)
            else:
                vectors = primal.fit_principal_component_regression(self.cov_X, self.tikhonov_reg, self.rank, self.svd_solver)
        self.U = vectors
        
        #Final Checks
        check_is_fitted(self, ['U', 'cov_XY', 'cov_X', 'cov_Y', 'X_fit', 'Y_fit'])
        self._is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None) \
            -> np.ndarray:
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial condition ``X``.
        
        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Parameters:
            X (numpy.ndarray): Initial conditions for which we wish the prediction, shape ``(n_init_conditions, n_features)``.
            t (int): Number of steps to predict (return the last one).
            observables (callable, numpy.ndarray or None): Callable, array of shape ``(n_samples, n_obs_features)`` or ``None``. If array, it must be the observable evaluated at ``self.Y_fit``. If ``None`` returns the predictions for the state.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, n_obs_features)``.
        """

        if observables is None:
            _obs = self.Y_fit
        elif callable(observables):
            _obs = observables(self.Y_fit)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "Observables must be either None, a callable or a Numpy array of the observable evaluated at the Y training points.")
        assert _obs.shape[0] == self.X_fit.shape[0]
        if _obs.ndim == 1:
            _obs = _obs[:, None]

        phi_Xin = self.feature_map(X)
        phi_X = self.feature_map(self.X_fit)
        return primal.predict(t, self.U, self.cov_XY, phi_Xin, phi_X, _obs)

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None) \
            -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Parameters:
            eval_left_on (numpy.ndarray or None): States of the system to evaluate the left eigenfunctions on, shape ``(n_samples, n_features)``.
            eval_right_on (numpy.ndarray or None): States of the system to evaluate the right eigenfunctions on, shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray or tuple: (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. Left eigenfunctions evaluated at ``eval_left_on``, shape ``(n_samples, rank)`` if ``eval_left_on`` is not ``None``. Right eigenfunction evaluated at ``eval_right_on``, shape ``(n_samples, rank)`` if ``eval_right_on`` is not ``None``.
        """

        check_is_fitted(self, ['U', 'cov_XY'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = primal.estimator_eig(self.U, self.cov_XY)
            self._eig_cache = (w, vl, vr)
        if eval_left_on is None and eval_right_on is None:
            return w
        elif eval_left_on is None and eval_right_on is not None:
            phi_Xin = self.feature_map(eval_right_on)
            return w, primal.evaluate_eigenfunction(phi_Xin, vr)
        elif eval_left_on is not None and eval_right_on is None:
            phi_Xin = self.feature_map(eval_left_on)
            return w, primal.evaluate_eigenfunction(phi_Xin, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            phi_Xin_l = self.feature_map(eval_left_on)
            phi_Xin_r = self.feature_map(eval_right_on)
            return w, primal.evaluate_eigenfunction(phi_Xin_l, vl), primal.evaluate_eigenfunction(phi_Xin_r, vr)
    
    def modes(self, X: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None) -> np.ndarray:
        """
        Computes the mode decomposition of the Koopman/Transfer operator of one or more observables of the system at the state ``X``.

        Parameters:
            X (numpy.ndarray): States of the system for which the modes are returned, shape ``(n_states, n_features)``.
            observables (callable, numpy.ndarray or None): Callable, array of shape ``(n_samples, n_obs_features)`` or ``None``. If array, it must be the observable evaluated at ``self.Y_fit``. If ``None`` returns the predictions for the state.

        Returns:
            numpy.ndarray: Modes of the system at the state ``X``, shape ``(self.rank, n_states, n_obs_features)``.
        """

        if observables is None:
            _obs = self.Y_fit
        elif callable(observables):
            _obs = observables(self.Y_fit)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated at the "
                "Y training points.")
        assert _obs.shape[0] == self.X_fit.shape[0]
        if _obs.ndim == 1:
            _obs = _obs[:, None]

        check_is_fitted(self, ['U', 'X_fit', 'cov_XY'])
        phi_X = self.feature_map(self.X_fit)
        phi_Xin = self.feature_map(X)
        _gamma = primal.estimator_modes(self.U, self.cov_XY, phi_X, phi_Xin)
        return np.squeeze(np.matmul(_gamma, _obs))  # [rank, num_initial_conditions, num_observables]

    def svals(self) -> np.ndarray:
        """Singular values of the Koopman/Transger operator.

        Returns:
            numpy.ndarray: The estimated singular values of the Koopman/Transfer operator, shape `(rank,)`.
        """        
        check_is_fitted(self, ['U', 'cov_XY'])
        return primal.svdvals(self.U, self.cov_XY)

    def save(self, path: os.PathLike):
        create_base_dir(path)
        with open(path, '+wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        with open(path, '+rb') as infile:
            restored_obj = pickle.load(infile)
            assert type(restored_obj) == cls
            return restored_obj

    def _init_covs(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the covariance matrices `cov_X`, `cov_Y`, and `cov_XY`.

        Args:
            X (np.ndarray): Training data of shape ``(n_samples, n_features)`` corresponding to the state at time t.
            Y (np.ndarray): Training data of shape ``(n_samples, n_features)`` corresponding to the state at time t+1.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - ``cov_X`` (np.ndarray): Covariance matrix of the feature map evaluated at X, shape ``(n_features, n_features)``.
                - ``cov_Y`` (np.ndarray): Covariance matrix of the feature map evaluated at Y, shape ``(n_features, n_features)``.
                - ``cov_XY`` (np.ndarray): Cross-covariance matrix of the feature map evaluated at X and Y, shape ``(n_features, n_features)``.
        """
        cov_X = self.feature_map.cov(X)
        cov_Y = self.feature_map.cov(Y)
        cov_XY = self.feature_map.cov(X, Y)
        return cov_X, cov_Y, cov_XY

    def _pre_fit_checks(self, X: np.ndarray, Y: np.ndarray):
        """Performs pre-fit checks on the training data.

        Use check_array and check_X_y from sklearn to check the training data, initialize the covariance matrices and
        save the training data.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.

        """
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        #TODO - check if the feature map is trainable, if so, check if it is fitted, if not, fit it.

        cov_X, cov_Y, cov_XY = self._init_covs(X, Y)

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

class DMD(ExtendedDMD):
    """
    Dynamic Mode Decomposition (DMD) Model.
    Implements the classical DMD estimator approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator. This model just a minimal wrapper around ``ExtendedDMD`` setting the feature map to the identity function. 
    """
    def __init__(self, 
                reduced_rank: bool = True,
                rank: Optional[int] = 5, 
                tikhonov_reg: float = 0,
                svd_solver: str = 'full',
                iterated_power: int = 1,
                n_oversamples: int = 5,
                rng_seed: Optional[int] = None):
        super().__init__(
            reduced_rank=reduced_rank, 
            rank=rank,
            tikhonov_reg=tikhonov_reg,
            svd_solver=svd_solver,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            rng_seed=rng_seed
        )