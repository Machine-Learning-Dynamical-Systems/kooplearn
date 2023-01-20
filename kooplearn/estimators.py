from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.extmath import randomized_svd

import numpy as np

from scipy.linalg import eig, eigh, LinAlgError, pinvh
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh, lsqr, cg
from scipy.sparse import diags
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv

from warnings import warn

from .utils import sort_and_crop, weighted_norm, modified_QR, SquaredKernel

class LowRankRegressor(BaseEstimator, RegressorMixin):
    def modes(self, observable = lambda x: x, _cached_results = None):
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

        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1..
            observable (lambda function, optional): Observable to forecast. Defaults to the identity map, corresponding to forecasting the state itself.
            which (None or array of integers, optional): If None, compute the forecast with all the modes of the observable. If which is an array of integers, the forecast is computed using only the modes corresponding to the indexes provided. The modes are arranged in decreasing order with respect to the eigenvalues. For example, if which = [0,2] only the first and third leading modes are used to forecast.  Defaults to None.

            This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).

            Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.

        Returns:
            ndarray: array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
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
            w (ndarray): Eigenvalues of the estimated Koopman Operator.
            fr (lambda function, only if right=True): Right eigenfunctions of the estimated Koopman Operator.
            fl (lambda function, only if left=True): Left eigenfunctions of the estimated Koopman Operator.
        """
        return self._eig(left=left, right=right, return_type = 'default')        
    def svals(self, k = 6, stabilizer = None):
        """Compute the largest singular values of Z = AS (A = Koopman operator, S = injection operator)

        Args:
            k (int, optional): Number of singular values to evaluate. Defaults to 6.
            stabilizer ([float, None], optional): A float to stabilize the inversion of K_x. It amounts to replaxe K_x -> K_x + stabilizer*Id. Defaults to None, corresponding to the pre-specified Tikhonov regularization in the RRR algorithm and to sqrt(num_points)^-1 in the PCR algorithm.

        Returns:
            NDArray: Array of the computed singular values decreasingly ordered.
        """
        check_is_fitted(self, ['K_X_', 'K_Y_'])
        
        dim = self.K_X_.shape[0]
        inv_dim = dim**(-1)
        if stabilizer is not None:
            alpha = stabilizer
        else:
            try:
                alpha = dim*self.tikhonov_reg
            except AttributeError:
                alpha = np.sqrt(dim) #dim*reg with reg~1/sqrt(dim)
        
        K = inv_dim*(self.K_Y_@self.K_X_)
        tikhonov = aslinearoperator(diags(np.ones(dim, dtype=self.K_X_.dtype)*alpha))
        _S = eigs(K, k, aslinearoperator(self.K_X_) + tikhonov, return_eigenvectors = False)
        assert np.max(np.abs(_S.imag)) < 1e-8, "The computed eigenvalues are not real. Possibly ill-conditioned problem"
        return np.flip(np.sort(_S.real))
    def _eig(self, left=False, right=True, return_type = 'default'):         
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        dim_inv = (self.K_X_.shape[0])**(-1)
        sqrt_inv_dim = dim_inv**0.5
        
        C = (self.V_.T)@((dim_inv * self.K_YX_)@(self.U_))

        w, vl, vr =  eig(C, left=True, right=True) #Left -> V, Right -> U

        if (not (left or right)) and (return_type == 'default'):
            return w

        # Sort the eigenvalues with respect to modulus 
        sortperm = np.argsort(w, kind='stable')[::-1]
        w = w[sortperm]
        vl = vl[:, sortperm]
        vr = vr[:, sortperm]

        W_X = self.U_.T @ ((self.K_X_ * dim_inv) @ self.U_)
        W_Y = self.V_.T @ ((self.K_Y_ * dim_inv) @ self.V_)
        
        norm_r = weighted_norm(vr,W_X)
        norm_l = weighted_norm(vl,W_Y)

        vr = vr @ np.diag(norm_r**(-1))
        vl = vl @ np.diag(norm_l**(-1))

        left_right_dot = np.sum(vl.conj()*vr, axis=0)
        
        fr = lambda X:  sqrt_inv_dim*self.kernel(X, self.X_fit_, backend=self.backend)@self.U_@vr
        fl = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y_fit_, backend=self.backend)@self.V_@vl    
        
        #If return_type != 'default', override the normal returns.
        if return_type == 'koopman_modes':
            left_right_norms = np.diag((w*left_right_dot)**-1)
            return w, left_right_norms, vl, fr
        elif return_type == 'eigenvalues_error_bounds':
            return w, vr
        else:    
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
        """Empirical risk of the model. :math:`\\frac{1}{n}\sum_{i = 1}^{n} ||\phi(Y_i) - G^*\phi(X_i)||^{2}_{\mathcal{H}}`

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
    def empirical_excess_risk(self, X, Y, norm = 'HS'):
        check_is_fitted(self, ['K_X_', 'K_Y_', 'U_', 'V_'])
        if norm == 'HS':
            return np.sqrt(self.risk(X = X, Y = Y))
        elif norm == 'op':
            X = np.asarray(self._validate_data(X=X, reset=True))
            Y = np.asarray(self._validate_data(X=Y, reset=True))

            #Everything processed on the numpy backend to avoid keops bugs
            K_v_Y = self.kernel(Y, backend = 'numpy')
            K_vt_Y = self.kernel(Y, self.Y_fit_, backend = 'numpy')
            K_tv_X = self.kernel(self.X_fit_, X, backend = 'numpy')

            val_dim, dim = K_vt_Y.shape
            sqrt_inv_dim = dim**(-0.5)
            V = sqrt_inv_dim*self.V_
            U = sqrt_inv_dim*self.U_
            
            C = K_vt_Y@V
            D = (K_tv_X.T@U).T
            E = (V.T)@(self.K_Y_@V)

            M = (val_dim**(-1))*(K_v_Y -(C@D + (D.T)@(C.T)) + ((D.T)@E@(D)))
            sigma_1_sq = np.max(np.linalg.eigvalsh(M))
            return np.sqrt(sigma_1_sq)
        else:
            raise ValueError(f"Accepted norms are 'HS' (Hilbert-Schmidt) or 'op' (Operator norm), while '{norm}' was provided.")
    def reconstruction_error(self, X, Y):
        Y_pred = self.predict(X)
        inv_dim = X.shape[0]**(-1.)
        return inv_dim*np.sum((Y - Y_pred)**2)
        
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
    def __init__(self, kernel=None, rank=5, tikhonov_reg=None, backend='numpy', svd_solver='full', iterated_power=1, n_oversamples=5, optimal_sketching=False):
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
        self.optimal_sketching = optimal_sketching
    
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

        if self.tikhonov_reg is None:
            U, V, sigma_sq = self._fit_unregularized(self.K_X_, self.K_Y_)
        else:
            U, V, sigma_sq = self._fit_regularized(self.K_X_, self.K_Y_)     
        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self
    
    def _fit_regularized(self, K_X, K_Y):
        dim = K_X.shape[0]
        inv_dim = dim**(-1)
        alpha = dim*self.tikhonov_reg
        norm_inducing_op = SquaredKernel(K_X, inv_dim, self.tikhonov_reg)
        if self.svd_solver =='randomized':
            tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K_X.dtype)*alpha))
            K_reg_inv = IterInv(aslinearoperator(K_X) + tikhonov, ifunc = cg)
            l = self.rank + self.n_oversamples
            if self.optimal_sketching:
                Cov = inv_dim*K_Y
                Om = np.random.multivariate_normal(np.zeros(dim, dtype=K_X.dtype), Cov, size=l).T
            else:
                Om = np.random.randn(dim, l)      
             
            for _ in range(self.iterated_power):
                #Powered randomized rangefinder
                Om = (inv_dim*K_Y)@(Om - alpha*K_reg_inv@Om)    
            KOm = K_reg_inv@Om
            KOmp = Om - alpha*KOm
            
            F_0 = (Om.T@KOmp)
            F_1 = (KOmp.T @ (inv_dim*(K_Y @ KOmp)))

            #Generation of matrices U and V.   
            try:
                sigma_sq, Q = eigh(F_1, F_0)
            except LinAlgError:  
                sigma_sq, Q = eig(pinvh(F_0)@F_1) 
            
            Q_norm = np.sum(Q.conj()*(F_0@Q), axis=0)
            Q = Q@np.diag(Q_norm**-0.5)
            _idxs = sort_and_crop(sigma_sq.real, self.rank)
            sigma_sq = sigma_sq.real
            
            Q = Q[:,_idxs] 
            U = (dim**0.5)*np.asfortranarray(KOm @ Q)
            V = (dim**0.5)*np.asfortranarray(KOmp @ Q)
            return U.real, V.real, sigma_sq
        else: # 'arnoldi' or 'full'
            K = inv_dim*(K_Y@K_X)
            #Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow. Prefer svd_solver == 'randomized' in such a case.
            if self.svd_solver == 'arnoldi':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K_X.dtype)*alpha))
                #Adding a small buffer to the Arnoldi-computed eigenvalues.
                sigma_sq, U = eigs(K, self.rank + 3, aslinearoperator(K_X) + tikhonov)  
            else: #'full'
                tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
                sigma_sq, U = eig(K, K_X + tikhonov)
            
            max_imag_part = np.max(U.imag)
            if max_imag_part >=2.2e-10:
                warn(f"The computed projector is not real. The Kernel matrix is severely ill-conditioned.")
            U = np.real(U)

            #Post-process U. Promote numerical stability via additional QR decoposition if necessary.
            U = U[:, sort_and_crop(sigma_sq.real, self.rank)]
            U, _, columns_permutation = modified_QR(U, M = norm_inducing_op, column_pivoting=True)
            U = U[:,np.argsort(columns_permutation)]
            if U.shape[1] < self.rank:
                warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). {self.rank - U.shape[1]} degrees of freedom will be ignored.")
                _zeroes = np.zeros((U.shape[0], self.rank - U.shape[1]))
                U = np.c_[U, _zeroes]
                assert U.shape[1] == self.rank
            
            V = K_X@np.asfortranarray(U)
            return U, V, sigma_sq

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
        return U, V, sigma_sq
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
            if self.backend == 'keops':
                raise NotImplementedError('Randomized SVD solver is not implemented with the Keops backend yet.')
            else:
                V, S, _ = randomized_svd(K_X, self.rank, n_oversamples=self.n_oversamples, n_iter=self.iterated_power, random_state=None)
            
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