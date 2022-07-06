from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import randomized_svd

import numpy as np

from scipy.linalg import eig, eigh, lstsq
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh, lsqr
from scipy.sparse import diags

from warnings import warn

from .sklearn_utils import sort_and_crop, weighted_norm, modified_QR, IterInv, SquaredKernel

class LowRankRegressor(RegressorMixin):
    def eig(self, left=False, right=True):
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_Y_', 'K_YX_', 'X_fit_', 'Y_fit_'])

        dim_inv = (self.K_X_.shape[0])**(-1)
        sqrt_inv_dim = dim_inv**0.5
        C = dim_inv * self.K_YX_@self.U_

        w, vl, vr =  eig(self.V_.T@C, left=True, right=True)

        if not (left or right):
            return w

        vl = self.V_@vl
        vr = self.U_@vr

        # Sort the eigenvalues with respect to modulus 
        sortperm = np.argsort(w)[::-1]
        w = w[sortperm]
        vl = vl[:, sortperm]
        vr = vr[:, sortperm]
        vr_cpy_ = vr.copy()
        
        norm_r = weighted_norm(vr,self.K_X_)*dim_inv
        norm_l = weighted_norm(vl._levecs,self.K_Y_)*dim_inv

        vr = np.asfortranarray(vr @ np.diag(norm_r**(-1)))
        vl = np.asfortranarray(vl @ np.diag(norm_l**(-1)))

        modes_to_invert_ = vr_cpy_ @np.diag(w*(norm_r**(-1)))
        fr = lambda X:  sqrt_inv_dim*self.kernel(X, self.X_fit_, backend=self.backend)@vr
        fl = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y_fit_, backend=self.backend)@vl    
        if left:
            if right:
                return w, fl, fr
            return w, fl
        return w, fr

    def predict():
        pass
    def score():
        pass

class ReducedRankRegression(BaseEstimator, LowRankRegressor):
    def __init__(self, kernel=None, rank=5, tikhonov_reg=None, backend='numpy', svd_solver='full', iterated_power=2, n_oversamples=10):
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
                If 'randomized', run randomized SVD by the method of Kostic, Novelli [add ref.]  
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

    def fit(self, X, Y):
        """Fit the Koopman operator estimator.
        Args:
            X (ndarray): Input observations.
            Y (ndarray): Evolved observations.
        Returns:
            self: Returns self.
        """
        self._check_backend_solver_compatibility()
        X = check_array(X, order='C', dtype=float, copy=True)
        Y = check_array(Y, order='C', dtype=float, copy=True)
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
            K_reg_inv = IterInv(K_X, alpha)
            #Utility function to solve linear systems
            l = self.rank + self.n_oversamples
            Omega = np.random.randn(dim,l)
            Omega = np.asfortranarray(Omega @ np.diag(np.linalg.norm(Omega,axis=0)**-1))
            #Rangefinder
            for pw in range(self.iterated_power):
                KyO = np.asfortranarray(K_Y@Omega)
                Omega = np.asfortranarray(KyO - alpha * K_reg_inv@KyO)
            Omega = K_reg_inv@np.asfortranarray(K_Y@Omega)
            #QR decomposition
            Q, _, columns_permutation = modified_QR(Omega, M = norm_inducing_op, column_pivoting=True)
            if self.rank > Q.shape[1]:
                warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). Reducing the rank to {Q.shape[1]}.")
                self.set_params(**{"rank": Q.shape[1]})

            #Generation of matrices U and V.    
            C = np.asfortranarray(K_X@np.asfortranarray(Q))
            sigma_sq, evecs = eigh(C.T @ (K_Y @ C))
            _idxs = sort_and_crop(sigma_sq, self.rank)
            sigma_sq = sigma_sq[_idxs]/(dim**2)
            evecs = evecs[:,_idxs]
            
            U = np.asfortranarray(Q @ evecs)
            V = K_X @ U
            return U, V
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
                    warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). Reducing the rank to {U.shape[1]}.")
                    self.set_params(**{"rank": U.shape[1]})
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
        for i in range(self.rank):
            U[:,i] = lsqr(K_X, V[:,i])[0] #Not optimal with this explicit loop
        return U, V

    def _init_kernels(self, X, Y):
        K_X = self.kernel(X, backend=self.backend)
        K_Y = self.kernel(Y, backend=self.backend)
        K_YX = self.kernel(Y, X, backend=self.backend)
        if self.backend == 'keops':
            K_X = aslinearoperator(K_X)
            K_Y = aslinearoperator(K_Y)
            K_YX = aslinearoperator(K_YX)
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

class PrincipalComponentRegression(BaseEstimator, LowRankRegressor):
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
                If 'randomized', run randomized SVD by the method of Kostic, Novelli [add ref.]  
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
        X = check_array(X, order='C', dtype=float, copy=True)
        Y = check_array(Y, order='C', dtype=float, copy=True)
        K_X, K_Y, K_YX = self._init_kernels(X, Y)
        dim = K_X.shape[0]

        self.K_X_ = K_X
        self.K_Y_ = K_Y
        self.K_YX_ = K_YX

        self.X_fit_ = X
        self.Y_fit_ = Y

        if self.svd_solver == 'arnoldi':
            S, V = eigsh(K_X, self.rank)
        elif self.svd_solver == 'full':
            S, V = eigh(K_X)
        else:
            if self.backend == 'keops':
                raise NotImplementedError('Randomized SVD solver is not implemented with the Keops backend yet.')
            else:
                V, S, _ = randomized_svd(K_X, self.rank, n_oversamples=self.n_oversamples, n_iter=self.iterated_power)
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
            V = V[:_test] *np.sqrt(dim) 
            U = V[:_test] @ np.diag(S[:_test]**-1)

            warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). Reducing the rank to {V.shape[1]}.")
            self.set_params(rank=V.shape[1]) 
           
        self.U_ = np.asfortranarray(U)
        self.V_ = np.asfortranarray(V)
        return self

    def _init_kernels(self, X, Y):
        K_X = self.kernel(X, backend=self.backend)
        K_Y = self.kernel(Y, backend=self.backend)
        K_YX = self.kernel(Y, X, backend=self.backend)
        if self.backend == 'keops':
            K_X = aslinearoperator(K_X)
            K_Y = aslinearoperator(K_Y)
            K_YX = aslinearoperator(K_YX)
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