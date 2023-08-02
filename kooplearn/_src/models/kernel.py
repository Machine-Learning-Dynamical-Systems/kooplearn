from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, Callable
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from kooplearn._src.kernels import BaseKernel, Linear
from kooplearn._src.operator_regression import dual
from kooplearn._src.models.abc import BaseModel
import pickle


class KernelLowRankRegressor(BaseModel, RegressorMixin):
    def __init__(
            self,
            kernel: BaseKernel = Linear(),
            rank: int = 5,
            tikhonov_reg: float = 0.,
            svd_solver: str = 'full',
            iterated_power: int = 1,
            n_oversamples: int = 5,
            optimal_sketching: bool = False,
            frac_inducing_points: float = 0.1
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
                If 'nystrom', use Nystrom estimators.
            iterated_power (int, optional): Number of iterations for the power method computed by solver = 'randomized'.
             Must be of range :math:`[0, \\infty)`. Defaults to 2, ignored if solver != 'randomized'.
            n_oversamples (int, optional): This parameter is only relevant when solver = 'randomized'. It corresponds
            to the additional number of random vectors to sample the range of X so as to ensure proper conditioning.
            Defaults to 10, ignored if solver != 'randomized'.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal
             sketching (computaitonally more expensive but more accurate).
             Defaults to False, ignored if solver != 'randomized'.
            frac_inducing_points (float, optional): Fraction of the training points to use as inducing points for the
            Nystrom estimator. Defaults to 0.1, ignored if solver != 'nystrom'.
        """

        # Initial checks
        if svd_solver not in ['full', 'arnoldi', 'randomized', 'nystrom']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if svd_solver == 'randomized' and iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if svd_solver == 'randomized' and n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        if svd_solver == 'nystrom' and frac_inducing_points < 0:
            raise ValueError('Invalid frac_inducing_points. Must be non-negative.')
        self.kernel = kernel
        self.frac_inducing_points = frac_inducing_points
        super(BaseModel).__init__(rank, tikhonov_reg, svd_solver, iterated_power, n_oversamples, optimal_sketching)
        self.K_YX_ = None
        self.K_Y_ = None
        self.K_X_ = None
        self.KN_Y_ = None
        self.KN_X_ = None

    def fit(self, X: ArrayLike, Y: ArrayLike):
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])

        if observables is None:
            _obs = self.Y_fit_
        elif callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated at the "
                "Y training points.")

        K_Xin_X = self.kernel(X, self.X_fit_)
        return dual.predict(t, self.U_, self.V_, self.K_YX_, K_Xin_X, _obs)

    def modes(self, Xin: ArrayLike, observables: Optional[Union[Callable, ArrayLike]] = None):
        if observables is None:
            _obs = self.Y_fit_
        elif callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated at the "
                "Y training points.")

        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        _, lv, rv = dual.estimator_eig(self.U_, self.V_, self.K_X_, self.K_YX_)
        K_Xin_X = self.kernel(Xin, self.X_fit_)
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
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(self.U_, self.V_, self.K_X_, self.K_YX_)
            self._eig_cache = (w, vl, vr)

        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                K_X_in_X_or_Y = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y, vr)
        else:
            if eval_right_on is None:
                K_X_in_X_or_Y = self.kernel(eval_left_on, self.Y_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y, vl)
            else:
                K_X_in_X_or_Y_left = self.kernel(eval_left_on, self.Y_fit_)
                K_X_in_X_or_Y_right = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y_left, vl), dual.evaluate_eigenfunction(
                    K_X_in_X_or_Y_right, vr)

    def svals(self):
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_'])
        return dual.svdvals(self.U_, self.V_, self.K_X_, self.K_Y_)

    def _init_kernels(self, X: ArrayLike, Y: ArrayLike):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX = self.kernel(Y, X)
        return K_X, K_Y, K_YX

    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        if self.svd_solver == 'nystrom':
            rng = np.random.default_rng()

            _data_dim = X.shape[0]
            _centers_idxs = rng.choice(_data_dim, int(_data_dim * self.frac_inducing_points))

            X_subset = X[_centers_idxs]
            Y_subset = Y[_centers_idxs]

            self.K_X_ = self.kernel(X_subset)
            self.K_Y_ = self.kernel(Y_subset)
            self.K_YX_ = self.kernel(Y_subset, X_subset)

            self.KN_X_ = self.kernel(X, X_subset)
            self.KN_Y_ = self.kernel(Y, Y_subset)

            self.X_fit_ = X_subset
            self.Y_fit_ = Y_subset

        else:
            K_X, K_Y, K_YX = self._init_kernels(X, Y)

            self.K_X_ = K_X
            self.K_Y_ = K_Y
            self.K_YX_ = K_YX

            self.X_fit_ = X
            self.Y_fit_ = Y
        if hasattr(self, '_eig_cache'):
            del self._eig_cache

    def _verify_adequacy(self, new_obj: KernelLowRankRegressor):
        if not hasattr(new_obj, 'kernel'):
            return False
        if self.frac_inducing_points != new_obj.frac_inducing_points:
            return False
        super(BaseModel)._verify_adequacy(new_obj)
        return True

    def load(self, filename, change_kernel=True):
        new_obj = super(BaseModel).load(filename)
        self.K_X_ = new_obj.K_X_.copy()
        self.K_Y_ = new_obj.K_Y_.copy()
        self.K_YX_ = new_obj.K_YX_.copy()
        if change_kernel:
            assert hasattr(new_obj, "kernel"), "savefile does not contain an kernel based model"
            self.kernel = new_obj.kernel


class KernelDMD(KernelLowRankRegressor):
    def fit(self, X, Y):
        """Fit the Koopman operator estimator. 
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        """
        self.pre_fit_checks(X, Y)

        if self.tikhonov_reg is None:
            reg = 0
        else:
            reg = self.tikhonov_reg

        if self.svd_solver == 'randomized':
            U, V = dual.fit_rand_tikhonov(self.K_X_, reg, self.rank, self.n_oversamples, self.iterated_power)
        elif self.svd_solver == 'nystrom':
            U, V = dual.fit_nystrom_tikhonov(self.K_X_, self.K_Y_, self.KN_X_, self.KN_Y_, reg)
            del self.KN_X_  # Free memory
            del self.KN_Y_  # Free memory
        else:
            U, V = dual.fit_tikhonov(self.K_X_, reg, self.rank, self.svd_solver)

        self.U_ = U
        self.V_ = V
        return self


class KernelReducedRank(KernelLowRankRegressor):
    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        """
        self.pre_fit_checks(X, Y)

        if self.svd_solver == 'nystrom':
            if self.tikhonov_reg is None:
                raise ValueError("tikhonov_reg must be specified when solver is nystrom.")
            else:
                U, V = dual.fit_nystrom_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.KN_X_, self.KN_Y_,
                                                                         self.tikhonov_reg, self.rank)
            del self.KN_X_  # Free memory
            del self.KN_Y_  # Free memory

        elif self.svd_solver == 'randomized':
            if self.tikhonov_reg is None:
                raise ValueError("tikhonov_reg must be specified when solver is randomized.")
            else:
                U, V = dual.fit_rand_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.tikhonov_reg,
                                                                      self.rank, self.n_oversamples,
                                                                      self.optimal_sketching, self.iterated_power)
        else:
            if self.tikhonov_reg is None:
                tikhonov_reg = 0
            else:
                tikhonov_reg = self.tikhonov_reg
            U, V = dual.fit_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, tikhonov_reg, self.rank,
                                                             self.svd_solver)
        self.U_ = U
        self.V_ = V
        return self
