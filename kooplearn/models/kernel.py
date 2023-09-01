from __future__ import annotations
import numpy as np
import os
import pickle
from pathlib import Path

from typing import Optional, Union, Callable
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from kooplearn._src.utils import check_is_fitted, create_base_dir
from sklearn.utils.validation import check_X_y
from sklearn.gaussian_process.kernels import Kernel, DotProduct
from kooplearn._src.operator_regression import dual
from kooplearn.abc import BaseModel

class KernelDMD(BaseModel, RegressorMixin):
    """
    Kernel Dynamic Mode Decomposition (KernelDMD) Model.
    Implements the KernelDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.

    Parameters:
        kernel (sklearn.gaussian_process.kernels.Kernel): sklearn Kernel object. Defaults to `DotProduct`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. Defaults to 5.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        optimal_sketching (bool): Sketching strategy for the randomized solver. If `True` performs optimal sketching (computaitonally expensive but more accurate).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.
    
    Attributes:
        X_fit : Training data of shape ``(n_samples, n_features)`` corresponding to a collection of sampled states.
        Y_fit : Evolved training data of shape ``(n_samples, n_features)`` corresponding the evolution of ``X_fit`` after one step.
        kernel_X : Kernel matrix of the states X_fit, shape ``(n_samples, n_samples)``.
        kernel_Y : Kernel matrix of the states Y_fit, shape ``(n_samples, n_samples)``.
        kernel_XY : Cross-kernel matrix of the states X_fit and Y_fit, shape ``(n_samples, n_samples)``.
        U : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).
        V : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).

    """
    def __init__(
            self,
            kernel: Kernel = DotProduct(),
            reduced_rank: bool = True,
            rank: int = 5,
            tikhonov_reg: Optional[float] = None,
            svd_solver: str = 'full',
            iterated_power: int = 1,
            n_oversamples: int = 5,
            optimal_sketching: bool = False,
            rng_seed: Optional[int] = None,
    ):
        # Initial checks
        if svd_solver not in ['full', 'arnoldi', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if svd_solver == 'randomized' and iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if svd_solver == 'randomized' and n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        
        self.rng_seed = rng_seed
        self.kernel = kernel
        self.rank = rank
        if tikhonov_reg is None:
            self.tikhonov_reg = 0.
        else:
            self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

        self.reduced_rank = reduced_rank
        self._is_fitted = False
        
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted    

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fits the KernelDMD model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model.

        Parameters:
            X : Training data of shape ``(n_samples, n_features)`` corresponding to a collection of sampled states.
            Y : Evolved training data of shape ``(n_samples, n_features)`` corresponding the evolution of ``X`` after one step.
        
        Returns:
            self (KernelDMD): The fitted estimator.
        """
        self.pre_fit_checks(X, Y)
        if self.reduced_rank:
            if self.svd_solver == 'randomized':
                if self.tikhonov_reg == 0.0:
                    raise ValueError("tikhonov_reg must be specified when solver is randomized.")
                else:
                    U, V = dual.fit_rand_reduced_rank_regression(self.kernel_X, self.kernel_Y, self.tikhonov_reg,self.rank, self.n_oversamples,self.optimal_sketching, self.iterated_power, rng_seed=self.rng_seed)
            else:
                U, V = dual.fit_reduced_rank_regression(self.kernel_X, self.kernel_Y, self.tikhonov_reg, self.rank, self.svd_solver)
        else:
            if self.svd_solver == 'randomized':
                U, V = dual.fit_rand_principal_component_regression(self.kernel_X, self.tikhonov_reg, self.rank, self.n_oversamples, self.iterated_power, rng_seed=self.rng_seed)
            else:
                U, V = dual.fit_principal_component_regression(self.kernel_X, self.tikhonov_reg, self.rank, self.svd_solver)
        self.U = U
        self.V = V

        #Final Checks
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', 'X_fit', 'Y_fit'])
        self._is_fitted = True
        return self   

    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None):
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
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', 'X_fit', 'Y_fit'])

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
        _obs_trailing_dims = _obs.shape[1:]
        expected_shape = (X.shape[0],) + _obs_trailing_dims
        if _obs.ndim > 2:
            _obs = _obs.reshape(_obs.shape[0], -1)

        K_Xin_X = self.kernel(X, self.X_fit)
        return dual.predict(t, self.U, self.V, self.kernel_YX, K_Xin_X, _obs).reshape(expected_shape)

    def modes(self, X: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        """
        Computes the mode decomposition of the Koopman/Transfer operator of one or more observables of the system at the state ``X``.

        Parameters:
            X (numpy.ndarray): States of the system for which the modes are returned, shape ``(n_states, n_features)``.
            observables (callable, numpy.ndarray or None): Callable, array of shape ``(n_samples, ...)`` or ``None``. If array, it must be the observable evaluated at ``self.Y_fit``. If ``None`` returns the predictions for the state.

        Returns:
            numpy.ndarray: Modes of the system at the state ``X``, shape ``(self.rank, n_states, ...)``.
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
        _obs_shape = _obs.shape
        if _obs.ndim > 2:
            _obs = _obs.reshape(_obs.shape[0], -1)

        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_YX', 'X_fit', 'Y_fit'])
        _, lv, rv = dual.estimator_eig(self.U, self.V, self.kernel_X, self.kernel_YX)
        K_Xin_X = self.kernel(X, self.X_fit)
        _gamma = dual.estimator_modes(K_Xin_X, rv, lv)

        expected_shape = (self.rank, X.shape[0], _obs_shape[1])
        return np.squeeze(np.matmul(_gamma, _obs).reshape(expected_shape))  # [rank, num_initial_conditions, num_observables]

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Parameters:
            eval_left_on (numpy.ndarray or None): States of the system to evaluate the left eigenfunctions on, shape ``(n_samples, n_features)``.
            eval_right_on (numpy.ndarray or None): States of the system to evaluate the right eigenfunctions on, shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray or tuple: (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. Left eigenfunctions evaluated at ``eval_left_on``, shape ``(n_samples, rank)`` if ``eval_left_on`` is not ``None``. Right eigenfunction evaluated at ``eval_right_on``, shape ``(n_samples, rank)`` if ``eval_right_on`` is not ``None``.
        """
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', 'X_fit', 'Y_fit'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(self.U, self.V, self.kernel_X, self.kernel_YX)
            self._eig_cache = (w, vl, vr)

        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                kernel_Xin_X_or_Y = self.kernel(eval_right_on, self.X_fit)
                return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr)
        else:
            if eval_right_on is None:
                kernel_Xin_X_or_Y = self.kernel(eval_left_on, self.Y_fit)
                return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vl)
            else:
                kernel_Xin_X_or_Y_left = self.kernel(eval_left_on, self.Y_fit)
                kernel_Xin_X_or_Y_right = self.kernel(eval_right_on, self.X_fit)
                return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_left, vl), dual.evaluate_eigenfunction(
                    kernel_Xin_X_or_Y_right, vr)

    def svals(self):
        """Singular values of the Koopman/Transger operator.

        Returns:
            numpy.ndarray: The estimated singular values of the Koopman/Transfer operator, shape `(rank,)`.
        """  
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y'])
        return dual.svdvals(self.U, self.V, self.kernel_X, self.kernel_Y)

    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX = self.kernel(Y, X)
        return K_X, K_Y, K_YX

    def pre_fit_checks(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        K_X, K_Y, K_YX = self._init_kernels(X, Y)

        self.kernel_X = K_X
        self.kernel_Y = K_Y
        self.kernel_YX = K_YX

        self.X_fit = X
        self.Y_fit = Y
        if hasattr(self, '_eig_cache'):
            del self._eig_cache

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