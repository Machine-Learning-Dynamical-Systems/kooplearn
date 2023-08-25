from __future__ import annotations
import numpy as np
import os
import pickle
from pathlib import Path
from numpy.typing import ArrayLike
from typing import Optional, Union, Callable
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from kooplearn._src.utils import check_is_fitted, create_base_dir
from sklearn.utils.validation import check_X_y
from sklearn.gaussian_process.kernels import Kernel, DotProduct
from kooplearn._src.operator_regression import dual
from kooplearn._src.models.abc import BaseModel

class KernelDMD(BaseModel, RegressorMixin):
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
        """Low rank Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the `kernels`
            submodule. Defaults to Linear.
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to 0.
            svd_solver (str, optional):
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the
                'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by solver = 'randomized'.
             Must be of range :math:`[0, \\infty)`. Defaults to 2, ignored if solver != 'randomized'.
            n_oversamples (int, optional): This parameter is only relevant when solver = 'randomized'. It corresponds
            to the additional number of random vectors to sample the range of X so as to ensure proper conditioning.
            Defaults to 10, ignored if solver != 'randomized'.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal
             sketching (computaitonally more expensive but more accurate).
             Defaults to False, ignored if solver != 'randomized'.
        """

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

    def fit(self, X: ArrayLike, Y: ArrayLike):
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

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
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

        K_Xin_X = self.kernel(X, self.X_fit)
        return dual.predict(t, self.U, self.V, self.kernel_YX, K_Xin_X, _obs)

    def modes(self, Xin: ArrayLike, observables: Optional[Union[Callable, ArrayLike]] = None):
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

        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_YX', 'X_fit', 'Y_fit'])
        _, lv, rv = dual.estimator_eig(self.U, self.V, self.kernel_X, self.kernel_YX)
        K_Xin_X = self.kernel(Xin, self.X_fit)
        _gamma = dual.estimator_modes(K_Xin_X, rv, lv)
        return np.squeeze(np.matmul(_gamma, _obs))  # [rank, num_initial_conditions, num_observables]

    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        """Eigenvalues and eigenvectors of the estimated Koopman operator.
        
        Args:
            eval_left_on (Optional[ArrayLike], optional): _description_. Defaults to None.
            eval_right_on (Optional[ArrayLike], optional): _description_. Defaults to None.

        Returns:
            tuple: (evals, fl, fr) where evals is an array of shape (self.rank,) containing the eigenvalues of the
            estimated Koopman operator, fl and fr are arrays containing the evaluation of the left and
            right eigenfunctions of the estimated Koopman operator on the data passed to the arguments eval_left_on
            and eval_right_on respectively.
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
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y'])
        return dual.svdvals(self.U, self.V, self.kernel_X, self.kernel_Y)

    def _init_kernels(self, X: ArrayLike, Y: ArrayLike):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX = self.kernel(Y, X)
        return K_X, K_Y, K_YX

    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
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