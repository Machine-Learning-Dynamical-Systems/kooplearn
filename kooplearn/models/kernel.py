import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from kooplearn._src.kernels import BaseKernel, Linear
from kooplearn._src import dual
from kooplearn.models.base import BaseModel

class KernelLowRankRegressor(BaseModel, RegressorMixin):
    def __init__(self, kernel: BaseKernel = Linear(), rank:int = 5, tikhonov_reg:Optional[float] = None, svd_solver: str = 'full', iterated_power:int = 1, n_oversamples:int =5, optimal_sketching:bool =False):
        """Low rank Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the `kernels` submodule. Defaults to None.
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to None.
            backend (str, optional): 
                If 'numpy' kernel matrices are formed explicitely and stored as numpy arrays.
                Defaults to 'numpy'.
            svd_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by svd_solver = 'randomized'. Must be of range :math:`[0, \infty)`. Defaults to 2.
            n_oversamples (int, optional): This parameter is only relevant when svd_solver = 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal sketching (computaitonally more expensive but more accurate). Defaults to False. (RRR only)
        """

        #Initial checks
        if svd_solver not in ['full', 'arnoldi', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if svd_solver == 'randomized' and iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if svd_solver == 'randomized' and n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
    
        self.kernel = kernel
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching
    
    def fit(self, X:ArrayLike, Y:ArrayLike):
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    def predict(self, X: ArrayLike, t:int = 1):
        """Predict an observable using the estimated Koopman operator.
        
        This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).
        Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.
        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            observables (ndarray, optional): 2D array of observables computed on previously seen data. If None, uses the test Y dataset instead.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1.
        Returns:
            ndarray: Array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        K_testX = self.kernel(X, self.X_fit_)
        if observables is None:
            return dual.low_rank_predict(t, self.U_, self.V_, self.K_YX_, K_testX, self.Y_fit_)
        return dual.low_rank_predict(t, self.U_, self.V_, self.K_YX_, K_testX, observables)

    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
        """Eigenvalues and eigenvectors of the estimated Koopman operator.
        

        Args:
            left (Optional[ArrayLike], optional): _description_. Defaults to None.
            right (Optional[ArrayLike], optional): _description_. Defaults to None.

        Returns:
            tuple: (evals, fl, fr) where evals is an array of shape (self.rank,) containing the eigenvalues of the estimated Koopman operator, fl and fr are arrays containing the evaluation of the left and right eigenfunctions of the estimated Koopman operator on the data passed to the arguments eval_left_on and eval_right_on respectively.
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        w, vl, vr  = dual.low_rank_eig(self.U_, self.V_, self.K_X_, self.K_Y_, self.K_YX_)
        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                K_testX = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.low_rank_eigfun_eval(K_testX, self.U_, vr)
        else:
            if eval_right_on is None:
                K_testX = self.kernel(eval_left_on, self.X_fit_)
                return w, dual.low_rank_eigfun_eval(K_testX, self.V_, vl)
            else:
                K_testX_left = self.kernel(eval_left_on, self.X_fit_)
                K_testX_right = self.kernel(eval_right_on, self.X_fit_)
                return w, dual.low_rank_eigfun_eval(K_testX_left, self.V_, vl), dual.low_rank_eigfun_eval(K_testX_right, self.U_, vr)
        
    def svals(self):
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_'])
        return dual.svdvals(self.U_, self.V_, self.K_X_, self.K_Y_)

    def _init_kernels(self, X: ArrayLike, Y: ArrayLike):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_XY = self.kernel(X,Y)
        return K_X, K_Y, K_XY
    
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

class KernelPrincipalComponent(KernelLowRankRegressor):
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
            U,V,_ = dual.fit_rand_tikhonov(self.K_X_, reg, self.rank, self.n_oversamples, self.iterated_power)
        else:
            U,V = dual.fit_tikhonov(self.K_X_, reg, self.rank, self.svd_solver)
            
        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self   

class KernelReducedRank(KernelLowRankRegressor):
    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        """
        self.pre_fit_checks(X, Y)
        if self.tikhonov_reg is None:
            U,V,_ = dual.fit_reduced_rank_regression_noreg(self.K_X_, self.K_Y_, self.rank, self.svd_solver)
        elif self.svd_solver == 'randomized':
            U,V,_ = dual.fit_rand_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.rank, self.tikhonov_reg, self.n_oversamples, self.optimal_sketching, self.iterated_power)
        else:
            U,V,_ = dual.fit_reduced_rank_regression_tikhonov(self.K_X_, self.K_Y_, self.rank, self.tikhonov_reg, self.svd_solver)

        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self