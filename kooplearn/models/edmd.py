from __future__ import annotations
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Optional, Callable, Union

from kooplearn._src.context_window_utils import check_contexts, contexts_to_markov_predict_states, contexts_to_markov_train_states
from kooplearn._src.utils import check_is_fitted, create_base_dir, enforce_2d_output
from kooplearn._src.linalg import cov
from kooplearn.abc import BaseModel, FeatureMap
from kooplearn.models.feature_maps import IdentityFeatureMap
from kooplearn._src.operator_regression import primal
import logging
logger = logging.getLogger('kooplearn')

class ExtendedDMD(BaseModel):
    """
    Extended Dynamic Mode Decomposition (ExtendedDMD) Model.
    Implements the ExtendedDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.
    
    Args:
        feature_map (callable): Dictionary of functions used for the ExtendedDMD algorithm. Should be a subclass of ``kooplearn.abc.FeatureMap``.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. ``None`` returns the full rank estimator.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    Attributes:
        data_fit : Training data. Array of context windows of shape ``(n_samples, context_len, *features_shape)`` corresponding to a collection of sampled states.
        cov_X : Covariance matrix of the feature map evaluated at X_fit.
        cov_Y : Covariance matrix of the feature map evaluated at Y_fit.
        cov_XY : Cross-covariance matrix of the feature map evaluated at X_fit and Y_fit.
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
        self._picklable_feature_map = feature_map
        self.feature_map = enforce_2d_output(feature_map)
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
    
    def fit(self, data: np.ndarray, lookback_len: Optional[int] = None) -> ExtendedDMD:
        """
        Fits the ExtendedDMD model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model. The feature map will act on arrays of shape ``(n_samples, lookback_len, *features_shape)``. The outputs of the feature map are **always** reshaped to 2D arrays of shape ``(n_samples, -1)``.

        .. warning::

            Extended DMD is an algorithm which uses lookforward windows of length strictly equal to 1. Therefore, if ``lookback_len`` is not ``None``, it must match ``lookback_len == context_len - 1``. Otherwise an error is raised.

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.
            lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None, corresponding to ``lookback_len = context_len - 1``.
        
        Returns:
            self (ExtendedDMD): The fitted estimator.
        """
        self._pre_fit_checks(data, lookback_len)
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
        check_is_fitted(self, ['U', 'cov_XY', 'cov_X', 'cov_Y', 'data_fit', '_lookback_len'])
        self._is_fitted = True
        return self
        
    def predict(self, data: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None) \
            -> np.ndarray:
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial condition ``X``.
        
        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Array of context windows with the same shape of the training data. The lookforward slice will be ignored, while the lookback slice defines the initial conditions from which we wish to predict, shape ``(n_init_conditions, context_len, *features_shape)``.
            t (int): Number of steps to predict (return the last one).
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_samples, context_len, *obs_features_shape)`` or ``None``. If array, it must be the observable evaluated at the training data. If ``None`` returns the predictions for the state.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, n_obs_features)``.
        """
        check_is_fitted(self, ['U', 'cov_XY', 'cov_X', 'cov_Y', 'data_fit', '_lookback_len'])
        
        #Shape checks:
        data = check_contexts(data, self._lookback_len, warn_len0_lookforward=True)
        if not ((data.shape[1] == self.data_fit.shape[1]) or (data.shape[1] == self._lookback_len)):
            raise ValueError(f"Shape mismatch between training data and inference data. The inference data has context length {data.shape[1]}, while the training data has context length {self.data_fit.shape[1]}.")

        X_inference, _ = contexts_to_markov_predict_states(data, self._lookback_len)
        X_fit, Y_fit = contexts_to_markov_predict_states(self.data_fit, self._lookback_len)

        if observables is None:
            _obs = Y_fit
        elif callable(observables):
            _obs = observables(Y_fit)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "Observables must be either None, a callable or a Numpy array of the observable evaluated at the Y training points.")
        
        assert _obs.shape[0] == Y_fit.shape[0], f"Observables have {_obs.shape[0]} samples while the number of training data is {Y_fit.shape[0]}."
        
        if _obs.ndim == 1:
            _obs = _obs[:, None]
        
        _obs_trailing_dims = _obs.shape[1:]
        expected_shape = (X_inference.shape[0],) + _obs_trailing_dims
        if _obs.ndim > 2:
            _obs = _obs.reshape(_obs.shape[0], -1)

        phi_Xin = self.feature_map(X_inference)
        phi_X = self.feature_map(X_fit)

        return (primal.predict(t, self.U, self.cov_XY, phi_Xin, phi_X, _obs)).reshape(expected_shape)

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

        check_is_fitted(self, ['U', 'cov_XY', '_lookback_len'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = primal.estimator_eig(self.U, self.cov_XY)
            self._eig_cache = (w, vl, vr)
        if eval_left_on is None and eval_right_on is None:
            return w
        elif eval_left_on is None and eval_right_on is not None:
            X, _ = contexts_to_markov_predict_states(eval_right_on, self._lookback_len)
            phi_Xin = self.feature_map(X)
            return w, primal.evaluate_eigenfunction(phi_Xin, vr)
        elif eval_left_on is not None and eval_right_on is None:
            X, _ = contexts_to_markov_predict_states(eval_left_on, self._lookback_len)
            phi_Xin = self.feature_map(X)
            return w, primal.evaluate_eigenfunction(phi_Xin, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            Xr, _ = contexts_to_markov_predict_states(eval_right_on, self._lookback_len)
            Xl, _ = contexts_to_markov_predict_states(eval_left_on, self._lookback_len)
            phi_Xin_l = self.feature_map(Xl)
            phi_Xin_r = self.feature_map(Xr)
            
            return w, primal.evaluate_eigenfunction(phi_Xin_l, vl), primal.evaluate_eigenfunction(phi_Xin_r, vr)
    
    def modes(self, data: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None) -> np.ndarray:
        """
        Computes the mode decomposition of the Koopman/Transfer operator of one or more observables of the system at the state ``X``.

        Args:
            data (numpy.ndarray): States of the system for which the modes are returned, shape ``(n_states, n_features)``.
            observables (callable, numpy.ndarray or None): Callable, array of shape ``(n_samples, ...)`` or ``None``. If array, it must be the observable evaluated at ``self.Y_fit``. If ``None`` returns the predictions for the state.

        Returns:
            numpy.ndarray: Modes of the system at the state ``X``, shape ``(self.rank, n_states, ...)``.
        """
        check_is_fitted(self, ['U', 'data_fit', 'cov_XY', '_lookback_len'])
        #Shape checks:
        data = check_contexts(data, self._lookback_len, warn_len0_lookforward=True)
        if not ((data.shape[1] == self.data_fit.shape[1]) or (data.shape[1] == self._lookback_len)):
            raise ValueError(f"Shape mismatch between training data and inference data. The inference data has context length {data.shape[1]}, while the training data has context length {self.data_fit.shape[1]}.")

        X_inference, _ = contexts_to_markov_predict_states(data, self._lookback_len)
        X_fit, Y_fit = contexts_to_markov_predict_states(self.data_fit, self._lookback_len)

        if observables is None:
            _obs = Y_fit
        elif callable(observables):
            _obs = observables(Y_fit)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated at the "
                "Y training points.")
        assert _obs.shape[0] == X_fit.shape[0], f"Observables have {_obs.shape[0]} samples while the number of training data is {Y_fit.shape[0]}."
        if _obs.ndim == 1:
            _obs = _obs[:, None]
        _obs_shape = _obs.shape
        if _obs.ndim > 2:
            _obs = _obs.reshape(_obs.shape[0], -1)

        phi_X = self.feature_map(X_fit)
        phi_Xin = self.feature_map(X_inference)

        _gamma = primal.estimator_modes(self.U, self.cov_XY, phi_X, phi_Xin)

        expected_shape = (self.rank, X_inference.shape[0]) + _obs_shape[1:]
        return np.squeeze(np.matmul(_gamma, _obs).reshape(expected_shape))  # [rank, num_initial_conditions, ...]

    def svals(self) -> np.ndarray:
        """Singular values of the Koopman/Transger operator.

        Returns:
            numpy.ndarray: The estimated singular values of the Koopman/Transfer operator, shape `(rank,)`.
        """        
        check_is_fitted(self, ['U', 'cov_XY'])
        return primal.svdvals(self.U, self.cov_XY)

    def save(self, path: os.PathLike):
        create_base_dir(path)
        del self.feature_map
        with open(path, '+wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        with open(path, '+rb') as infile:
            restored_obj = pickle.load(infile)
            assert type(restored_obj) == cls
            restored_obj.feature_map = enforce_2d_output(restored_obj._picklable_feature_map)
            return restored_obj

    def _init_covs(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the covariance matrices `cov_X`, `cov_Y`, and `cov_XY`.

        Args:
            stacked (np.ndarray): Training data of shape ``(n_samples, 2,  *features_shape)``. It should be the result of the function :func:`stack_lookback`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - ``cov_X`` (np.ndarray): Covariance matrix of the feature map evaluated at X, shape ``(n_features, n_features)``.
                - ``cov_Y`` (np.ndarray): Covariance matrix of the feature map evaluated at Y, shape ``(n_features, n_features)``.
                - ``cov_XY`` (np.ndarray): Cross-covariance matrix of the feature map evaluated at X and Y, shape ``(n_features, n_features)``.
        """
        X = self.feature_map(X)
        Y = self.feature_map(Y)

        cov_X = cov(X)
        cov_Y = cov(Y)
        cov_XY = cov(X, Y)
        return cov_X, cov_Y, cov_XY

    def _pre_fit_checks(self, data: np.ndarray, lookback_len: Optional[int] = None) -> None:
        """Performs pre-fit checks on the training data.

        Use :func:`check_contexts` to check and sanitize the input data, initialize the covariance matrices and saves the training data.

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.
            lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None, corresponding to ``lookback_len = context_len - 1``.

        """
        if lookback_len is None:
            lookback_len = data.shape[1] - 1
        data = check_contexts(data, lookback_len, enforce_len1_lookforward=True)
        if hasattr(self, '_lookback_len'):
            del self._lookback_len
        #Save the lookback length as a private attribute of the model
        self._lookback_len = lookback_len
        X_fit, Y_fit = contexts_to_markov_train_states(data, self._lookback_len)

        self.cov_X, self.cov_Y, self.cov_XY = self._init_covs(X_fit, Y_fit)
        self.data_fit = data

        if self.rank is None:
            self.rank = min(self.cov_X.shape[0], self.data_fit.shape[0])
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