from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

import numpy as np

from scipy.linalg import eig, eigh, lstsq, solve
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh, lsqr
from scipy.sparse import diags

from warnings import warn

from .utils import sort_and_crop, weighted_norm, modified_QR, IterInv, SquaredKernel

class KernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=None, tikhonov_reg = None, backend='numpy', num_modes = None):
        """Initialize the Koopman operator estimator.
        
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the ``kernels``submodule. Defaults to None corresponds to a linear kernel.
            tikhonov_reg (_type_, optional): Tikhonov regularization parameter. Defaults to None.
            backend (str, optional): 
                If 'numpy' kernel matrices are formed explicitely and stored as numpy arrays. 
                If 'keops', kernel matrices are computed on the fly and never stored in memory. Keops backend is GPU compatible and preferable for large scale problems. 
                Defaults to 'numpy'.
            eig_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                Defaults to 'full'.
            num_modes (int, optional):
                Number of modes to compute. Defaults to None. If num_modes is None, all modes are computed calling LAPACK solver functions. If num_modes is an integer, the modes corresponding to the first num_modes leading eigenvalues are computed calling ARPACK solver functions.
        """
        self.kernel = kernel
        self.tikhonov_reg = tikhonov_reg
        self.backend = backend
        if np.isscalar(num_modes):
            self.num_modes = int(num_modes)
        else:
            self.num_modes = num_modes
    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        Returns:
            self: Returns self.
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

        self.n_features_in_ = X.shape[1]
        return self  
    def modes(self, observable = lambda x: x, _modes_to_invert = None):
        """Modes of the estimated Koopman operator.

        Args:
            observable (lambda function, optional): _description_. Defaults to the identity map, corresponding to computing the modes of the state itself.
            _modes_to_invert (ndarray, optional): Internal parameter used if cached results can be exploited. Defaults to None.

        Returns:
            ndarray: Array of shape (self.rank, n_obs) containing the estimated modes of the observable(s) provided as argument. Here n_obs = len(observable(x)).
        """        
        check_is_fitted(self, ['V_', 'K_X_', 'Y_fit_'])
        inv_sqrt_dim = (self.K_X_.shape[0])**(-0.5)
        evaluated_observable = observable(self.Y_fit_)
        if evaluated_observable.ndim == 1:
            evaluated_observable = evaluated_observable[:,None]
        if _modes_to_invert is None:
            _, _, _modes_to_invert = self._eig(_return_modes_to_invert=True) #[TODO] This could be cached for improved performance
        modes = lstsq(_modes_to_invert, evaluated_observable)[0] 
        return modes*inv_sqrt_dim           
    def forecast(self, X, t=1., observable = lambda x: x, which = None,):
        """Forecast an observable using the estimated Koopman operator.

        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1..
            observable (lambda function, optional): Observable to forecast. Defaults to the identity map, corresponding to forecasting the state itself.
            which (None or array of integers, optional): If None, compute the forecast with all the modes of the observable. If which is an array of integers, the forecast is computed using only the modes corresponding to the indexes provided. The modes are arranged in decreasing order with respect to the eigenvalues. For example, if which = [0,2] only the first and third leading modes are used to forecast.  Defaults to None.

            This method with t=1., observable = lambda x: x, self.num_modes == None and which = None is equivalent to self.predict(X).

        Returns:
            ndarray: array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """        
        check_is_fitted(self, ['K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        evals, _refuns, modes_to_invert = self._eig(_return_modes_to_invert=True)
        modes = self.modes(self, observable=observable, _modes_to_invert=modes_to_invert)
        
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
            w (ndarray): Eigenvalues of the estimated Koopman Operator.
            fr (lambda function, only if right=True): Right eigenfunctions of the estimated Koopman Operator.
            fl (lambda function, only if left=True): Left eigenfunctions of the estimated Koopman Operator. Returned only if self.num_modes == None.
        """
        return self._eig(left=left, right=right, _return_modes_to_invert=False)
    def _eig(self, left=False, right=True, _return_modes_to_invert = False):         
        check_is_fitted(self, ['K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        dim_inv = (self.K_X_.shape[0])**(-1)
        dim = self.K_X_.shape[0]
        sqrt_inv_dim = dim_inv**0.5
        
        if self.tikhonov_reg is not None:
            M = self.K_X_ + dim*self.tikhonov_reg*np.eye(dim)
            A = self.K_YX_
        else:
            M = None
            A = np.linalg.pinv(self.K_X_)@self.K_YX_ #Not efficient but works
        
        if not (left or right):
            if self.num_modes is not None:
                return eigs(A, k=self.num_modes, M=M, which='LM', return_eigenvectors=False)
            else:
                return eig(A, M, left=left, right=right)
        
        if left:
            assert self.num_modes is None, "If left=True, num_modes must be None. Unable to return left eigenfunctions from ARPACK routines."
            if right:
                w, vl, vr = eig(A, M, left=left, right=right)
            else:
                w, vl = eig(A, M, left=left, right=right)
                vr = None
        else:
            #Only right eigenfunctions are returned
            if self.num_modes is not None:
                w, vr = eigs(A, k=self.num_modes, M=M, which='LM', return_eigenvectors=True)
            else:
                w, vr = eig(A, M, left=left, right=right)
            vl = None

        # Sort the eigenvalues with respect to modulus 
        sortperm = np.argsort(w)[::-1]
        w = w[sortperm]
        
        if vl is not None:
            vl = vl[:, sortperm]
            norm_l = weighted_norm(vl,self.K_Y_)*dim_inv
            vl = vl @ np.diag(norm_l**(-0.5))
            fl = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y_fit_, backend=self.backend)@vl
        
        if vr is not None:
            vr = vr[:, sortperm]
            norm_r = weighted_norm(vr,self.K_X_)*dim_inv
            vr = vr @ np.diag(norm_r**(-0.5))
            fr = lambda X:  sqrt_inv_dim*self.kernel(X, self.X_fit_, backend=self.backend)@vr
            modes_to_invert_ = self.K_YX_@vr * dim_inv

        #If _return_modes_to_invert is True, override the normal returns.
        if _return_modes_to_invert:
            return w, fr, modes_to_invert_
        if left:
            if right:
                return w, fl, fr
            return w, fl
        else:
            return w, fr
    def predict(self, X):
        """Predict the state of the system at the next step using the estimated Koopman operator.
        Args:
            X (ndarray): Array of shape (n_samples, n_features) containing n_samples states of the system.

        Returns:
            ndarray: Array of shape (n_samples, n_features) containing the one-step-ahead prediction of the states in X.
        """
        check_is_fitted(self, ["K_X_", "Y_fit_"])
        X = np.asarray(self._validate_data(X=X, reset=True))
        dim = self.X_fit_.shape[0]
        if self.tikhonov_reg is not None:
            tikhonov = np.eye(dim)*(self.tikhonov_reg*dim)
            _Z = solve(self.K_X_ + tikhonov,self.Y_fit_, assume_a='pos')      
        else:
            _Z = lstsq(self.K_X_, self.Y_fit_)[0]
        
        onedim_state_space = self.X_fit_.shape[1] == 1
        if X.ndim == 1 and onedim_state_space: #Many samples with one feature
            X = X[:, None]
        elif X.ndim == 1 and (not onedim_state_space): #One sample with many features
            X = X[None,:]
        elif np.ndim(X) == 0: #One sample with one feature
            X = np.asarray(X)[None, None]
        _S = self.kernel(X, self.X_fit_, backend = self.backend)
        return _S@_Z
    def _init_kernels(self, X, Y):
        K_X = self.kernel(X, backend=self.backend)
        K_Y = self.kernel(Y, backend=self.backend)
        K_YX = self.kernel(Y, X, backend=self.backend)
        return K_X, K_Y, K_YX
    def _check_backend_solver_compatibility(self):
        if self.backend not in ['numpy', 'keops']:
            raise ValueError('Invalid backend. Allowed values are \'numpy\' and \'keops\'.')
        if self.num_modes is not None:
            if self.num_modes < 1:
                raise ValueError('num_modes must be an integer greater than 0.')
        if self.backend == 'keops':
            raise NotImplementedError("Keops backend is not yet implemented.")
        return
    def _init_risk(self, X, Y):
        check_is_fitted(self, ['K_X_', 'K_Y_', 'X_fit_', 'Y_fit_'])
        if (X is not None) and (Y is not None):
            X = np.asarray(self._validate_data(X=X, reset=True))
            Y = np.asarray(self._validate_data(X=Y, reset=True))
            K_yY = self.kernel(Y, self.Y_fit_, backend = self.backend)
            K_Xx = self.kernel(self.X_fit_, X, backend = self.backend)
            _Y = Y
        else:
            K_yY = self.K_Y_
            K_Xx = self.K_X_
            _Y = self.Y_fit_
        r_yy = 0
        for y in _Y:
            y = y[None,:]
            r_yy += self.kernel(y,y, backend='numpy')
        r_yy = np.squeeze(r_yy)*((_Y.shape[0])**(-1))             
        return K_yY, K_Xx, r_yy
    def risk(self, X = None, Y = None):
        """Empirical risk of the model. n^{-1}\sum_{i = 1}^{n} ||\phi(Y_i) - G^*\phi(X_i)||^{2}_{H}

        Args:
            X (ndarray, optional): Array of shape (num_test_points, num_features) of input observations. Defaults to None.
            Y (ndarray, optional): Array of shape (num_test_points, num_features) of evolved observations. Defaults to None.
            If X == Y == None, the traning data is used. And the sample training risk is returned.

        Returns:
            float: Risk of the Kernel Ridge Regression estimator.
        """
        check_is_fitted(self, ['K_X_'])        
        K_yY, K_Xx, r = self._init_risk(X, Y)
        val_dim, dim = K_yY.shape
        if self.tikhonov_reg is not None:
            tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
            C = solve(self.K_X_ + tikhonov, K_Xx, assume_a='pos')
        else:
            C = lstsq(self.K_X_, K_Xx)[0]
        r -= 2*(val_dim**(-1))*np.trace(K_yY@C)
        r += (val_dim**(-1))*np.trace(C.T@(self.K_Y@C))
        return r        
    def _more_tags(self):
        return {
            'multioutput_only': True,
            'non_deterministic': True,
            'poor_score': True,
            "_xfail_checks": {
                # check_estimator checks that fail:
                "check_dict_unchanged": "Comparing ndarrays (input data + kernel) with == fails. Could be fixed by using np.allclose.",
                }
            }
class LowRankRegressor(BaseEstimator, RegressorMixin):
    def modes(self, observable = lambda x: x, _modes_to_invert = None):
        """Modes of the estimated Koopman operator.

        Args:
            observable (lambda function, optional): _description_. Defaults to the identity map, corresponding to computing the modes of the state itself.
            _modes_to_invert (ndarray, optional): Internal parameter used if cached results can be exploited. Defaults to None.

        Returns:
            ndarray: Array of shape (self.rank, n_obs) containing the estimated modes of the observable(s) provided as argument. Here n_obs = len(observable(x)).
        """        
        check_is_fitted(self, ['V_', 'K_X_', 'Y_fit_'])
        inv_sqrt_dim = (self.K_X_.shape[0])**(-0.5)
        evaluated_observable = observable(self.Y_fit_)
        if evaluated_observable.ndim == 1:
            evaluated_observable = evaluated_observable[:,None]
        evaluated_observable = (self.V_.T)@evaluated_observable
        if _modes_to_invert is None:
            _, _, _modes_to_invert = self._eig(_return_modes_to_invert=True) #[TODO] This could be cached for improved performance
        modes = lstsq(_modes_to_invert, evaluated_observable)[0] 
        return modes*inv_sqrt_dim        
    def forecast(self, X, t=1., observable = lambda x: x, which = None,):
        """Forecast an observable using the estimated Koopman operator.

        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1..
            observable (lambda function, optional): Observable to forecast. Defaults to the identity map, corresponding to forecasting the state itself.
            which (None or array of integers, optional): If None, compute the forecast with all the modes of the observable. If which is an array of integers, the forecast is computed using only the modes corresponding to the indexes provided. The modes are arranged in decreasing order with respect to the eigenvalues. For example, if which = [0,2] only the first and third leading modes are used to forecast.  Defaults to None.

            This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).

        Returns:
            ndarray: array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        evals, _refuns, modes_to_invert = self._eig(_return_modes_to_invert=True)
        modes = self.modes(observable=observable, _modes_to_invert=modes_to_invert)
        
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
            w (ndarray): Eigenvalues of the estimated Koopman Operator.
            fr (lambda function, only if right=True): Right eigenfunctions of the estimated Koopman Operator.
            fl (lambda function, only if left=True): Left eigenfunctions of the estimated Koopman Operator.
        """
        return self._eig(left=left, right=right, _return_modes_to_invert=False)
    def _eig(self, left=False, right=True, _return_modes_to_invert = False):         
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        dim_inv = (self.K_X_.shape[0])**(-1)
        sqrt_inv_dim = dim_inv**0.5
        C = dim_inv * self.K_YX_@self.U_

        w, vl, vr =  eig(self.V_.T@C, left=True, right=True)

        if not (left or right):
            return w
        
        vr_cpy_ = vr.copy()

        vl = self.V_@vl
        vr = self.U_@vr

        # Sort the eigenvalues with respect to modulus 
        sortperm = np.argsort(w)[::-1]
        w = w[sortperm]
        vl = vl[:, sortperm]
        vr = vr[:, sortperm]
        vr_cpy_ = vr_cpy_[:, sortperm]
        
        
        norm_r = weighted_norm(vr,self.K_X_)*dim_inv
        norm_l = weighted_norm(vl,self.K_Y_)*dim_inv

        vr = np.asfortranarray(vr @ np.diag(norm_r**(-1)))
        vl = np.asfortranarray(vl @ np.diag(norm_l**(-1)))

        modes_to_invert_ = vr_cpy_ @np.diag(w*(norm_r**(-1)))

        
        fr = lambda X:  sqrt_inv_dim*self.kernel(X, self.X_fit_, backend=self.backend)@vr
        fl = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y_fit_, backend=self.backend)@vl    
        
        #If _return_modes_to_invert is True, override the normal returns.
        if _return_modes_to_invert:
            return w, fr, modes_to_invert_
        if left:
            if right:
                return w, fl, fr
            return w, fl
        else:
            return w, fr
    def predict(self, X):
        """Predict the state of the system at the next step using the estimated Koopman operator.
        Args:
            X (ndarray): Array of shape (n_samples, n_features) containing n_samples states of the system.

        Returns:
            ndarray: Array of shape (n_samples, n_features) containing the one-step-ahead prediction of the states in X.
        """
        check_is_fitted(self, ["U_", "V_", "X_fit_", "Y_fit_"])
        X = np.asarray(self._validate_data(X=X, reset=True))
        
        sqrt_dim_inv = (self.K_X_.shape[0])**(-0.5)
        _Z = sqrt_dim_inv * self.V_.T @ self.Y_fit_
        onedim_state_space = self.X_fit_.shape[1] == 1
        if X.ndim == 1 and onedim_state_space: #Many samples with one feature
            X = X[:, None]
        elif X.ndim == 1 and (not onedim_state_space): #One sample with many features
            X = X[None,:]
        elif np.ndim(X) == 0: #One sample with one feature
            X = np.asarray(X)[None, None]
        _init_K = self.kernel(X, self.X_fit_, backend = self.backend)
        _S = sqrt_dim_inv * _init_K@self.U_
        return _S@_Z 
    def _init_kernels(self, X, Y):
        K_X = self.kernel(X, backend=self.backend)
        K_Y = self.kernel(Y, backend=self.backend)
        K_YX = self.kernel(Y, X, backend=self.backend)
        return K_X, K_Y, K_YX    
    def _check_backend_solver_compatibility(self):
        if self.backend not in ['numpy', 'keops']:
            raise ValueError('Invalid backend. Allowed values are \'numpy\' and \'keops\'.')
        if self.svd_solver not in ['full', 'arnoldi', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if self.svd_solver == 'randomized' and self.iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if self.svd_solver == 'randomized' and self.n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        if self.svd_solver == 'full' and self.backend == 'keops':
            raise ValueError('Invalid backend and svd_solver combination. \'keops\' backend is not compatible with \'full\' svd_solver.')
        return
    def _init_risk(self, X, Y):
        check_is_fitted(self, ['K_X_', 'K_Y_', 'X_fit_', 'Y_fit_'])
        if (X is not None) and (Y is not None):
            X = np.asarray(self._validate_data(X=X, reset=True))
            Y = np.asarray(self._validate_data(X=Y, reset=True))
            K_yY = self.kernel(Y, self.Y_fit_, backend = self.backend)
            K_Xx = self.kernel(self.X_fit_, X, backend = self.backend)
            _Y = Y
        else:
            K_yY = self.K_Y_
            K_Xx = self.K_X_
            _Y = self.Y_fit_
        r_yy = 0
        for y in _Y:
            y = y[None,:]
            r_yy += self.kernel(y,y, backend='numpy')
        r_yy = np.squeeze(r_yy)*((_Y.shape[0])**(-1))             
        return K_yY, K_Xx, r_yy
    def risk(self, X = None, Y = None):
        """Empirical risk of the model. n^{-1}\sum_{i = 1}^{n} ||\phi(Y_i) - G^*\phi(X_i)||^{2}_{H}

        Args:
            X (ndarray, optional): Array of shape (num_test_points, num_features) of input observations. Defaults to None.
            Y (ndarray, optional): Array of shape (num_test_points, num_features) of evolved observations. Defaults to None.
            If X == Y == None, the traning data is used. And the sample training risk is returned.

        Returns:
            float: Risk of the Low Rank Regression estimator.
        """
        check_is_fitted(self, ['K_X_', 'K_Y_', 'U_', 'V_'])        

        K_yY, K_Xx, r = self._init_risk(X, Y)
        val_dim, dim = K_yY.shape
        sqrt_inv_dim = dim**(-0.5)
        V = sqrt_inv_dim*self.V_
        U = sqrt_inv_dim*self.U_

        C = K_yY@V
        D = (K_Xx.T@U).T
        E = (V.T)@(self.K_Y_@V)

        r -= 2*(val_dim**(-1))*np.trace(D@C)
        r += (val_dim**(-1))*np.trace(E@D@D.T)
        return r
    def norm(self):
        """
            Hilbert-Schmidt norm of the estimator ||\hat{S}^{*}@U@V.T@\hat{Z}||_{HS} = n^{-2}tr(U.T@K_X@U@V.T@K_Y@V)
        """
        check_is_fitted(self, ['K_X_', 'K_Y_', 'U_', 'V_'])
        dim = self.K_X_.shape[0]
        sqrt_inv_dim = dim**(-0.5)
        U = sqrt_inv_dim*self.U_
        V = sqrt_inv_dim*self.V_
        U_X = U.T@(self.K_X_@U)
        V_Y = V.T@(self.K_Y_@V)
        return np.sqrt(np.trace(U_X@V_Y))
        
    def _more_tags(self):
        return {
            'multioutput_only': True,
            'non_deterministic': True,
            'poor_score': True,
            "_xfail_checks": {
                # check_estimator checks that fail:
                "check_dict_unchanged": "Comparing ndarrays (input data + kernel) with == fails. Could be fixed by using np.allclose.",
                }
            }
class ReducedRank(LowRankRegressor):
    def __init__(self, kernel=None, rank=5, tikhonov_reg=None, backend='numpy', svd_solver='full', iterated_power=2, n_oversamples=10, override_checks=False):
        """Reduced Rank Regression Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the ``kernels``submodule. Defaults to None corresponds to a linear kernel.
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
        self.tikhonov_reg = tikhonov_reg
        self.backend = backend
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.override_checks = override_checks
    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        Returns:
            self: Returns self.
        """
        self._check_backend_solver_compatibility()
        if not self.override_checks:
            X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
            Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
            check_X_y(X, Y, multi_output=True)

        K_X, K_Y, K_YX = self._init_kernels(X, Y)

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

        if self.tikhonov_reg is None:
            U, V = self._fit_unregularized(self.K_X_, self.K_Y_)
        else:
            U, V = self._fit_regularized(self.K_X_, self.K_Y_)     
        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self
    def _fit_regularized(self, K_X, K_Y):
        dim = K_X.shape[0]
        inv_dim = dim**(-1)
        alpha = dim*self.tikhonov_reg
        norm_inducing_op = SquaredKernel(K_X, inv_dim, self.tikhonov_reg)
        if self.svd_solver =='randomized':
            raise NotImplementedError()
        else: # 'arnoldi' or 'full'
            K = inv_dim*(K_Y@K_X)
            #Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow. Prefer svd_solver == 'randomized' in such a case.
            if self.svd_solver == 'arnoldi':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K_X.dtype)*alpha))
                Minv = IterInv(K_X, alpha)
                sigma_sq, U = eigs(K, self.rank, aslinearoperator(K_X) + tikhonov, Minv = Minv)  
            else: #'full'
                tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
                sigma_sq, U = eig(K, K_X + tikhonov)
            
            #Post-process U. Promote numerical stability via additional QR decoposition if necessary.
            U = U[:, sort_and_crop(sigma_sq, self.rank)]

            #Check that the eigenvectors are real
            if np.max(np.abs(np.sin(np.angle((U))))) > 1e-8:
                warn("Computed projector is not real. The Kernel matrix is either severely ill conditioned or non-symmetric, discarting imaginary parts.")
                #[TODO] Actually, the projector might be ok and complex if a global phase is present. Fix this.
            U = np.real(U)

            #Orthogonalize through pivoted QR algorithm
            U_norms = weighted_norm(U, M = norm_inducing_op)
            max_U_norm = np.max(U_norms)

            if any(U_norms < max_U_norm * 2.2e-16):  #Columns of U are too close to be linearly dependent. Perform QR factorization with pivoting to expose rank deficiency.
                U, _, columns_permutation = modified_QR(U, M = norm_inducing_op, column_pivoting=True)
                U = U[:,np.argsort(columns_permutation)]
                if U.shape[1] < self.rank:
                    warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). {self.rank - U.shape[1]} degrees of freedom will be ignored.")
                    _zeroes = np.zeros((U.shape[0], self.rank - U.shape[1]))
                    U = np.c_[U, _zeroes]
                    assert U.shape[1] == self.rank
            else:
                U = U@np.diag(U_norms**-1) 
            V = K_X@np.asfortranarray(U)
            return U, V
    def _fit_unregularized(self, K_X, K_Y):
        if self.svd_solver == 'randomized':
            warn("The 'randomized' svd_solver is equivalent to 'arnoldi' when tikhonov_reg = None.")
        #Solve the Hermitian eigenvalue problem to find V
        if self.svd_solver != 'full':
            sigma_sq, V = eigsh(K_Y, self.rank)
        else:
            sigma_sq, V = eigh(K_Y)
            V = V[:, sort_and_crop(sigma_sq, self.rank)]
        
        #Normalize V
        _V_norm = np.linalg.norm(V,ord=2,axis=0)/np.sqrt(V.shape[0])
        V = V@np.diag(_V_norm**-1)

        #Solve the least squares problem to determine U
        U = np.zeros_like(V)
        for i in range(U.shape[1]):
            U[:,i] = lsqr(K_X, V[:,i])[0] #Not optimal with this explicit loop
        return U, V
class PrincipalComponent(LowRankRegressor):
    def __init__(self, kernel=None, rank=5, backend='numpy', svd_solver='full', iterated_power=2, n_oversamples=10):
        """Reduced Rank Regression Estimator for the Koopman Operator
        Args:
            kernel (Kernel, optional): Kernel object implemented according to the specification found in the ``kernels``submodule. Defaults to None corresponds to a linear kernel.
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
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        Returns:
            self: Returns self.
        """
        self._check_backend_solver_compatibility()
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)
    
        K_X, K_Y, K_YX = self._init_kernels(X, Y)
        
        dim = K_X.shape[0]

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

        self.n_features_in_ = X.shape[1]

        if self.svd_solver == 'arnoldi':
            S, V = eigsh(K_X, self.rank)
        elif self.svd_solver == 'full':
            S, V = eigh(K_X)
        else:
            raise NotImplementedError('Randomized SVD solver is not implemented with the Keops backend yet.')
            
        sigma_sq = S**2
        sort_perm = sort_and_crop(sigma_sq, self.rank)   
        sigma_sq = sigma_sq[sort_perm]
        V = V[:,sort_perm]
        S = S[sort_perm]
        
        _test = S>2.2e-16
        if all(_test):            
            V = V * np.sqrt(dim) 
            U = V @ np.diag(S**-1)
        else:
            V = V[:,_test] *np.sqrt(dim) 
            U = V @ np.diag(S[_test]**-1)

            warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). {self.rank - V.shape[1]} degrees of freedom will be ignored.")
            _zeroes = np.zeros((V.shape[0], self.rank - V.shape[1]))
            V = np.c_[V, _zeroes]
            U = np.c_[U, _zeroes]
            assert U.shape[1] == self.rank
            assert V.shape[1] == self.rank       
           
        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self