from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from BaseKoopmanEstimator import BaseKoopmanEstimator
from _algorithms import dual

import numpy as np

class KernelLowRankRegressor(BaseKoopmanEstimator, RegressorMixin):

    def modes(self, observable=lambda x : x, _cached_results=None):
        """Modes of the estimated Koopman operator.

        Args:
            observable (lambda function, optional): _description_. Defaults to the identity map, corresponding to computing the modes of the state itself.
            _modes_to_invert (ndarray, optional): Internal parameter used if cached results can be exploited. Defaults to None.
        
        Returns:
            ndarray: Array of shape (self.rank, n_obs) containing the estimated modes of the observable(s) provided as argument. Here n_obs = len(observable(x)).
        """        
        check_is_fitted(self, ['V_', 'K_X_', 'Y_fit_'])
        inv_sqrt_dim = (self.K_X_.shape[0])**(-0.5)
        evaluated_observable = observable(self.Y_fit_).T
        if evaluated_observable.ndim == 1:
            evaluated_observable = evaluated_observable[None, :]
        if _cached_results is None:
            _, left_right_norms, vl, _ = self._eig(return_type='koopman_modes')
        else:
            (left_right_norms, vl) = _cached_results
        
        modes = evaluated_observable@self.V_@vl.conj()@left_right_norms
        return modes.T*inv_sqrt_dim

    def forecast(self, X, t=1., observable = lambda x: x, which = None,):
        """Forecast an observable using the estimated Koopman operator.
        
        This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).
        Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.
        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1.
            observable (lambda function, optional): Observable to forecast. Defaults to the identity map, corresponding to forecasting the state itself.
            which (None or array of integers, optional): If None, compute the forecast with all the modes of the observable. If which is an array of integers, the forecast is computed using only the modes corresponding to the indexes provided. The modes are arranged in decreasing order with respect to the eigenvalues. For example, if which = [0,2] only the first and third leading modes are used to forecast.  Defaults to None.
        Returns:
            ndarray: Array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        if len(X.shape) == 1:
            X = X.reshape(1,-1)

        evals, left_right_norms, vl, _refuns = self._eig(return_type='koopman_modes')
        cached_results = (left_right_norms, vl)
        modes = self.modes(observable=observable, _cached_results = cached_results)
        
        if which is not None:
            evals = evals[which][:, None]           # [r,1]
            refuns = _refuns(X)[:,which]            # [n,r]
            modes = modes[which,:]                  # [r,n_obs]
        else:
            evals = evals[:, None]                  # [r,1]
            refuns = _refuns(X)                     # [n,r]

        if np.isscalar(t):
            t = np.array([t], dtype=np.float64)[None,:] # [1, t]
        elif np.ndim(t) == 1:
            t = np.array(t, dtype=np.float64)[None,:]   # [1, t]
        else:
            raise ValueError("t must be a scalar or a 1D array.")
        evals_t = np.power(evals, t) # [r,t]
        forecasted = np.einsum('ro,rt,nr->tno', modes, evals_t, refuns)  # [t,n,n_obs]
        if forecasted.shape[0] <= 1:
            return np.real(forecasted[0])
        else:
            return np.real(forecasted)

    def eig(self, left=False, right=True):
        """Eigenvalue decomposition of the estimated Koopman operator.
        Args:
            left (bool, optional): Whether to return the left eigenfunctions. Defaults to False.
            right (bool, optional): Wheter to return the right eigenfunctions. Defaults to True.
        Returns:
            tuple: (evals, fr, fl) where evals is an array of shape (self.rank,) containing the eigenvalues of the estimated Koopman operator, fr is a lambda function returning the right eigenfunctions of the estimated Koopman operator, and fl is a lambda function returning the left eigenfunctions of the estimated Koopman operator. If left=False, fl is not returned. If right=False, fr is not returned.
        """
        return self._eig(left=left, right=right, return_type = 'default')  

    def _init_kernels(self, X, Y):
        K_X = self.kernel(X, backend=self.backend)
        K_Y = self.kernel(Y, backend=self.backend)
        K_XY = self.kernel(X,Y, backend=self.backend)
        return K_X, K_Y, K_XY

class KernelPrincipalComponent(KernelLowRankRegressor):
    def __init__(self, kernel=None, rank=5, backend='numpy', svd_solver='full', iterated_power=2, n_oversamples=10):
        """Principal Component Regression Estimator for the Koopman Operator
        
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the `kernels` submodule. Defaults to None corresponds to a linear kernel.
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (_type_, optional): Tikhonov regularization parameter. Defaults to None.
            backend (str, optional): 
                If 'numpy' kernel matrices are formed explicitely and stored as numpy arrays. 
                If 'keops', kernel matrices are computed on the fly and never stored in memory. Keops backend is GPU compatible and preferable for large scale problems. 
                Defaults to 'numpy'.
            svd_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by svd_solver == 'randomized'. Must be of range [0, infinity). Defaults to 2.
            n_oversamples (int, optional): This parameter is only relevant when svd_solver == 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10.
        """
        self.kernel = kernel
        self.rank = rank
        self.backend = backend
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples

    def fit(self, X, Y):
        # TODO
        self._check_backend_solver_compatibility()
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        K_X, K_Y, K_YX = self._init_kernels(X, Y)

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

    def predict(self, X, t=1, observable=lambda x : x):
        # TODO
        pass

class KernelReducedRank(KernelLowRankRegressor):

    def __init__(self, kernel=None, rank=5, tikhonov_reg=None, backend='numpy', svd_solver='full', iterated_power=1, n_oversamples=5, optimal_sketching=False):
        """Reduced Rank Regression Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the `kernels` submodule. Defaults to None.
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to None.
            backend (str, optional): 
                If 'numpy' kernel matrices are formed explicitely and stored as numpy arrays. 
                If 'keops', kernel matrices are computed on the fly and never stored in memory. Keops backend is GPU compatible and preferable for large scale problems. 
                Defaults to 'numpy'.
            svd_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by svd_solver = 'randomized'. Must be of range :math:`[0, \infty)`. Defaults to 2.
            n_oversamples (int, optional): This parameter is only relevant when svd_solver = 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal sketching (computaitonally more expensive but more accurate). Defaults to False.
        """
        self.kernel = kernel
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.backend = backend
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        """
        self._check_backend_solver_compatibility()
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        K_X, K_Y, K_YX = self._init_kernels(X, Y)

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

        if self.tikhonov_reg == None:
            U, V, sigma_eq = dual.fit_reduced_rank_regression_noreg(K_X, K_Y, self.rank, self.svd_solver)
        elif self.svd_solver == 'randomized':
            U,V,sigma_eq = dual.fit_rand_reduced_rank_regression_tikhonov(K_X, K_Y, self.rank, self.tikhonov_reg, self.n_oversamples, self.optimal_sketching, self.iterated_power)
        else:
            U,V,sigma_eq = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, self.rank, self.tikhonov_reg, self.svd_solver)

        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self

    def predict(self, X, t=1, observable=lambda x : x):
        # TODO
        pass