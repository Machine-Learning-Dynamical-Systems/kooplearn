import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, Callable
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from kooplearn._src.kernels import BaseKernel, Linear
from kooplearn._src.operator_regression import dual
from kooplearn.models.base import BaseModel

class KernelLowRankRegressor(BaseModel, RegressorMixin):
    def __init__(
            self, 
            kernel: BaseKernel = Linear(), 
            rank:int = 5, 
            tikhonov_reg:float = 0., 
            solver: str = 'full', 
            iterated_power:int = 1, 
            n_oversamples:int = 5, 
            optimal_sketching:bool = False,
            frac_inducing_points:float = 0.1
        ):
        """Low rank Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the `kernels` submodule. Defaults to Linear.
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to 0.
            solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
                If 'nystrom', use Nystrom estimators.
            iterated_power (int, optional): Number of iterations for the power method computed by solver = 'randomized'. Must be of range :math:`[0, \infty)`. Defaults to 2, igonred if solver != 'randomized'.
            n_oversamples (int, optional): This parameter is only relevant when solver = 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10, ignored if solver != 'randomized'.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal sketching (computaitonally more expensive but more accurate). Defaults to False, ignored if solver != 'randomized'.
            frac_inducing_points (float, optional): Fraction of the training points to use as inducing points for the Nystrom estimator. Defaults to 0.1, ignored if solver != 'nystrom'.
        """

        #Initial checks
        if solver not in ['full', 'arnoldi', 'randomized', 'nystrom']:
            raise ValueError('Invalid solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if solver == 'randomized' and iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if solver == 'randomized' and n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        if solver == 'nystrom' and frac_inducing_points < 0:
            raise ValueError('Invalid frac_inducing_points. Must be non-negative.')
        
        self.kernel = kernel
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.solver = solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching
        self.frac_inducing_points = frac_inducing_points
    
    def fit(self, X:ArrayLike, Y:ArrayLike):
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    def predict(self, X: ArrayLike, t:int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):   
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])

        if observables is None:
            _obs = self.Y_fit_
        elif callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError("observables must be either None, a callable or a Numpy array of the observable evaluated at the Y training points.")
        
        K_Xin_X = self.kernel(X, self.X_fit_)
        return dual.predict(t, self.U_, self.V_, self.K_YX_, K_Xin_X, _obs)

    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
        """Eigenvalues and eigenvectors of the estimated Koopman operator.
        

        Args:
            left (Optional[ArrayLike], optional): _description_. Defaults to None.
            right (Optional[ArrayLike], optional): _description_. Defaults to None.

        Returns:
            tuple: (evals, fl, fr) where evals is an array of shape (self.rank,) containing the eigenvalues of the estimated Koopman operator, fl and fr are arrays containing the evaluation of the left and right eigenfunctions of the estimated Koopman operator on the data passed to the arguments eval_left_on and eval_right_on respectively.
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        w, vl, vr  = dual.estimator_eig(self.U_, self.V_, self.K_X_, self.K_Y_, self.K_YX_)
        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                K_X_in_X_or_Y = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y, self.U_, vr)
        else:
            if eval_right_on is None:
                K_X_in_X_or_Y = self.kernel(eval_left_on, self.Y_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y, self.V_, vl)
            else:
                K_X_in_X_or_Y_left = self.kernel(eval_left_on, self.Y_fit_)
                K_X_in_X_or_Y_right = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.evaluate_eigenfunction(K_X_in_X_or_Y_left, self.V_, vl), dual.evaluate_eigenfunction(K_X_in_X_or_Y_right, self.U_, vr)
        
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
    
        K_X, K_Y, K_YX = self._init_kernels(X, Y)

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

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
        
        if self.solver == 'randomized':
            U,V = dual.fit_rand_tikhonov(self.K_X_, reg, self.rank, self.n_oversamples, self.iterated_power)
        else:
            U,V = dual.fit_tikhonov(self.K_X_, reg, self.rank, self.solver)
            
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
        if self.solver == 'nystrom':
            raise NotImplementedError()
        if self.tikhonov_reg is None:
            if self.solver == 'randomized':
                raise ValueError("tikhonov_reg must be specified when solver is randomized.")
            else:
                U,V = dual.fit_reduced_rank_regression(self.K_X_, self.K_Y_, self.rank, self.solver)
        else:
            if self.solver == 'randomized':
                U,V = dual.fit_rand_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.tikhonov_reg, self.rank, self.n_oversamples, self.optimal_sketching, self.iterated_power)
            else:
                U,V = dual.fit_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.tikhonov_reg, self.rank, self.svd_solver)
        self.U_ = U
        self.V_ = V
        return self