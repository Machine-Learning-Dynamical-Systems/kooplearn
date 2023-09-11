from __future__ import annotations
import numpy as np
import os
import pickle
from pathlib import Path

from typing import Optional, Union, Callable
from sklearn.base import RegressorMixin
from kooplearn._src.context_window_utils import check_contexts, contexts_to_markov_predict_states, contexts_to_markov_train_states
from kooplearn._src.utils import check_is_fitted, create_base_dir, enforce_2d_inputs, _parse_DMD_observables
from sklearn.gaussian_process.kernels import Kernel, DotProduct
from kooplearn._src.operator_regression import dual
from kooplearn.abc import BaseModel

class KernelDMD(BaseModel, RegressorMixin):
    """
    Kernel Dynamic Mode Decomposition (KernelDMD) Model.
    Implements the KernelDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.

    Args:
        kernel (sklearn.gaussian_process.kernels.Kernel): sklearn Kernel object. Defaults to `DotProduct`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. Defaults to 5.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        optimal_sketching (bool): Sketching strategy for the randomized solver. If `True` performs optimal sketching (computaitonally expensive but more accurate).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.
    
    .. tip::

        When ``lookback_len > 1``, the snapshots in the lookback window are first passed through the feature map and the results are then flattened and concatenated together. Concatenating multiple consecutive snapshots to form DMD estimators is a technique known as Hankel-DMD, originally proposed by :footcite:t:`Arbabi2017`. The covariance matrices are then computed on the resulting array of shape ``(n_samples, lookback_len*prod(features_shape))``.
    
    Attributes:
        data_fit : Training data: array of context windows of shape ``(n_samples, context_len, *features_shape)``.
        kernel_X : Kernel matrix evaluated at the initial states, that is ``data_fit[:, :lookback_len, ...]`. Shape ``(n_samples, n_samples)``
        kernel_Y : Kernel matrix evaluated at the evolved states, that is ``data_fit[:, 1:lookback_len + 1, ...]``. Shape ``(n_samples, n_samples)``
        kernel_XY : Cross-kernel matrix between initial and evolved states. Shape ``(n_samples, n_samples)``.
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
        self._picklable_kernel = kernel
        self.kernel = enforce_2d_inputs(kernel)
        if not isinstance(rank, int) or rank < 1:
            raise ValueError('rank must be a positive integer.')
        
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

    def fit(self, data: np.ndarray, lookback_len: Optional[int] = None) -> KernelDMD:
        """
        Fits the KernelDMD model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model. 
        
        
        .. attention::

            Internally, the data will be reshaped to ``(n_samples, lookback_len, lookback_len*prod(features_shape))`` before computing the kernel matrices. This is particularly important when ``lookback_len > 1``. See the note below for more details.

        .. attention::

            Extended DMD is an algorithm which uses lookforward windows of length strictly equal to 1. Therefore, if ``lookback_len`` is not ``None``, it must match ``lookback_len == context_len - 1``. Otherwise an error is raised.

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.
            lookback_len (Optional[int], optional): Length of the lookback window associated to the contexts. Defaults to None, corresponding to ``lookback_len = context_len - 1``.
        
        Returns:
            The fitted estimator.
        """
        self._pre_fit_checks(data, lookback_len)
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

    def predict(self, data: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None) \
            -> np.ndarray:
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``X = data[:, self._lookback_len:, ...]`` being the lookback slice of ``data``. 
        
        .. attention::
            
            ``data.shape[1]`` must either match the lookback length declared upon calling ``fit(data, lookback_len)`` or the context length of the training data ``self.data_fit.shape[1]``. Otherwise, an error is raised.
        
        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Initial conditions to predict. Array of context windows with shape ``(n_init_conditions, *self.data_fit.shape[1:])`` or ``(n_init_conditions, self._lookback_len, *self.data_fit.shape[2:])`` (see the note above).
            t (int): Number of steps in the future to predict (returns the last one).
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(self.data_fit.shape[0], self.data_fit.shape[1], *obs_features_shape)`` or ``None``. If array, it must be the desired observable evaluated on the *lookforward slice* of the training data. If ``None`` returns the predictions for the state.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, *obs_features_shape)``.
        """
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', 'data_fit', '_lookback_len'])

        _obs, expected_shape, X_inference, X_fit = _parse_DMD_observables(observables, data, self.data_fit, self._lookback_len)

        K_Xin_X = self.kernel(X_inference, X_fit)
        return dual.predict(t, self.U, self.V, self.kernel_YX, K_Xin_X, _obs).reshape(expected_shape)

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (numpy.ndarray or None): Array of context windows on which the left eigenfunctions are evaluated, shape ``(n_samples, *self.data_fit.shape[1:])``.
            eval_right_on (numpy.ndarray or None): Array of context windows on which the right eigenfunctions are evaluated, shape ``(n_samples, *self.data_fit.shape[1:])``.

        Returns:
            (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``  are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``: shape ``(n_samples, rank)``.
        """

        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', '_lookback_len', 'data_fit'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(self.U, self.V, self.kernel_X, self.kernel_YX)
            self._eig_cache = (w, vl, vr)
        
        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, self._lookback_len)
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            X, _ = contexts_to_markov_predict_states(eval_right_on, self._lookback_len)
            kernel_Xin_X_or_Y = self.kernel(X, X_fit)
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr)
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            X, _ = contexts_to_markov_predict_states(eval_left_on, self._lookback_len)
            kernel_Xin_X_or_Y = self.kernel(X, Y_fit)
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)

            Xr, _ = contexts_to_markov_predict_states(eval_right_on, self._lookback_len)
            Xl, _ = contexts_to_markov_predict_states(eval_left_on, self._lookback_len)

            kernel_Xin_X_or_Y_left = self.kernel(Xl, Y_fit)
            kernel_Xin_X_or_Y_right = self.kernel(Xr, X_fit)
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_left, vl), dual.evaluate_eigenfunction(
                kernel_Xin_X_or_Y_right, vr)
            
    def modes(self, data: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None) -> np.ndarray:
        """
        Computes the mode decomposition of arbitrary observables of the Koopman/Transfer operator at the states defined by ``data``.
        
        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        Args:
            data (numpy.ndarray): Initial conditions to compute the modes on. See :func:`predict` for additional details.
            observables (callable, numpy.ndarray or None): Callable, array of context windows of shape ``(n_samples, self.data_fit.shape[1], *obs_features_shape)`` or ``None``. If array, it must be the desired observable evaluated on the *lookforward slice* of the training data. If ``None`` returns the predictions for the state.

        Returns:
            Modes of the system at the states defined by ``data``. Array of shape ``(self.rank, n_states, ...)``.
        """
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_YX', '_lookback_len', 'data_fit'])
        
        _obs, expected_shape, X_inference, X_fit = _parse_DMD_observables(observables, data, self.data_fit, self._lookback_len)

        if hasattr(self, '_eig_cache'):
            _, lv, rv = self._eig_cache
        else:
            _, lv, rv = dual.estimator_eig(self.U, self.V, self.kernel_X, self.kernel_YX)
        
        K_Xin_X = self.kernel(X_inference, X_fit)
        _gamma = dual.estimator_modes(K_Xin_X, rv, lv)

        expected_shape = (self.rank,) + expected_shape
        return np.squeeze(np.matmul(_gamma, _obs).reshape(expected_shape))  # [rank, num_initial_conditions, num_observables]

    def svals(self):
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(rank,)`.
        """  
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y'])
        return dual.svdvals(self.U, self.V, self.kernel_X, self.kernel_Y)

    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX = self.kernel(Y, X)
        return K_X, K_Y, K_YX

    def _pre_fit_checks(self, data: np.ndarray, lookback_len: Optional[int] = None) -> None:
        """Performs pre-fit checks on the training data.

        Use :func:`check_contexts` to check and sanitize the input data, initialize the kernel matrices and saves the training data.

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

        self.kernel_X, self.kernel_Y, self.kernel_YX = self._init_kernels(X_fit, Y_fit)
        self.data_fit = data

        if hasattr(self, '_eig_cache'):
            del self._eig_cache

    def save(self, path: os.PathLike):
        create_base_dir(path)
        del self.kernel
        with open(path, '+wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        with open(path, '+rb') as infile:
            restored_obj = pickle.load(infile)
            assert type(restored_obj) == cls
            restored_obj.kernel = enforce_2d_inputs(restored_obj._picklable_kernel)
            return restored_obj