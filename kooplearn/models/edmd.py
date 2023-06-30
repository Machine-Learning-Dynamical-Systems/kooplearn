import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from kooplearn.models.base import BaseModel
from kooplearn._src import primal
from kooplearn._src.encoding_decoding_utils import FeatureMap, IdentityFeatureMap


class PrimalRegressor(BaseModel):
    def __init__(self, feature_map: FeatureMap = IdentityFeatureMap(), rank=5, tikhonov_reg=None, svd_solver='full', iterated_power=1, n_oversamples=5, optimal_sketching=False):
        """Reduced Rank Regression Estimator for the Koopman Operator
        Args:
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to None.
        
            svd_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by svd_solver = 'randomized'. Must be of range :math:`[0, \infty)`. Defaults to 2.
            n_oversamples (int, optional): This parameter is only relevant when svd_solver = 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal sketching (computaitonally more expensive but more accurate). Defaults to False.
        """
        self.feature_map = feature_map
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        """Predict an observable using the estimated Koopman operator.
        
        This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).
        Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.
        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            observables (ndarray, optional): 2D array of observables computed on previously seen data. If None, uses the test Y dataset instead.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1.
            observables (Optional[Union[Callable, ArrayLike]], optional): Observable(s) to forecast. If None, predicts the state itself. Defaults to None.
        Returns:
            ndarray: Array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """ 
        if observables is None:
            _obs = self.Y_fit_
        if callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError("observables must be either None, a callable or a Numpy array of the observable evaluated at the Y training points.")
        
        phi_testX = self.feature_map(X)
        phi_trainX = self.feature_map(self.X_fit_)
        return primal.low_rank_predict(t, self.U_, self.C_XY_, phi_testX, phi_trainX, _obs)
    
    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
        """Eigenvalues and eigenvectors of the estimated Koopman operator.
        

        Args:
            left (Optional[ArrayLike], optional): _description_. Defaults to None.
            right (Optional[ArrayLike], optional): _description_. Defaults to None.

        Returns:
            tuple: (evals, fl, fr) where evals is an array of shape (self.rank,) containing the eigenvalues of the estimated Koopman operator, fl and fr are arrays containing the evaluation of the left and right eigenfunctions of the estimated Koopman operator on the data passed to the arguments eval_left_on and eval_right_on respectively.
        """        
        check_is_fitted(self, ['U_','C_XY_'])
        w, vr  = primal.low_rank_eig(self.U_, self.C_XY_)
        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                phi_textX = self.feature_map(eval_right_on)
                return w, primal.low_rank_eigfun_eval(phi_textX, vr)
        else:
            raise NotImplementedError("Left eigenfunctions are not implemented yet.")

    def svd(self):
        check_is_fitted(self, ['U_', 'C_XY_'])
        return primal.svdvals(self.U_, self.C_XY_)
    
    def _init_covs(self, X: ArrayLike, Y: ArrayLike):
        C_X = self.feature_map.cov(X)
        C_Y = self.feature_map.cov(Y)
        C_XY = self.feature_map.cov(X,Y)
        return C_X, C_Y, C_XY
        
    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        C_X, C_Y, C_XY = self._init_covs(X, Y)

        self.C_X_ = C_X
        self.C_Y_ = C_Y
        self.C_XY_ = C_XY

        self.X_fit_ = X
        self.Y_fit_ = Y

class EDMDReducedRank(PrimalRegressor):
    def fit(self, X, Y):
        self.pre_fit_checks(X, Y)
        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.n_oversamples, self.iterated_power)
        else:
            vectors = primal.fit_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.svd_solver)
        self.U_ = vectors

class EDMD(PrimalRegressor):
    def fit(self, X, Y):
        self.pre_fit_checks(X, Y)
        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.n_oversamples, self.iterated_power)
        else:
            vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.svd_solver)
        self.U_ = vectors