from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

import numpy as np

from scipy.linalg import eig, lstsq, solve
from scipy.sparse.linalg import eigs

from ..utils import weighted_norm
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
        sortperm = np.argsort(w, kind='stable')[::-1]
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
        """Empirical risk of the model. :math:`\\frac{1}{n}\sum_{i = 1}^{n} ||\phi(Y_i) - G^*\phi(X_i)||^{2}_{\mathcal{H}}`

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