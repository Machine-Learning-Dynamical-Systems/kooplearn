from abc import ABCMeta, abstractmethod
from bdb import effective
import numpy as np
from scipy.linalg import eig, eigh, solve, lstsq
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh, lsqr
from scipy.sparse import diags
from .utils import modified_norm_sq, parse_backend, IterInv, KernelSquared, _check_real, modified_QR
from warnings import warn

class KoopmanRegression(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, Y, backend='auto'):
        """
            For low-rank estimators, compute the matrices U and V.
            For high-rank estimators, pass.
        """
        pass

    def modes(self, observable = lambda x: x):
        """
            Compute the modes associated to the given observable (should be a callable).
        """
        try:
            inv_sqrt_dim = (self.K_X.shape[0])**(-0.5)
            evaluated_observable = observable(self.Y)
            if evaluated_observable.ndim == 1:
                evaluated_observable = evaluated_observable[:,None]
            
            if isinstance(self, LowRankKoopmanRegression):
                evaluated_observable = (self.V.T)@evaluated_observable
            
            self._modes, _, effective_rank, _ = lstsq(self._modes_to_invert, evaluated_observable) 
            self._modes *= inv_sqrt_dim
            # if not isinstance(self, LowRankKoopmanRegression):
            #     self._modes = np.diag(self._evals) @ self._modes
            self._modes_observable = observable  
            return self._modes
        except AttributeError:
            try:
                self.eig()
                return self.modes(observable)
            except AttributeError:
                raise AttributeError("You must first fit the model.")
    
    def forecast(self, X, t=1., which = None):
        try:
            evaluated_observable = self._modes_observable(X)
            if evaluated_observable.ndim == 1:
                evaluated_observable = evaluated_observable[:,None]

            if which is not None:
                evals = self._evals[which][:, None]     # [r,1]
                refuns = self._refuns(evaluated_observable)[:,which]       # [n,r]
                modes = self._modes[which,:]            # [r,n_obs]
            else:
                evals = self._evals[:, None]                # [r,1]
                refuns = self._refuns(evaluated_observable)                # [n,r]
                modes = self._modes                     # [r,n_obs]

            if np.isscalar(t):
                t = np.array([t], dtype=np.float64)[None,:] # [1, t]
            elif np.ndim(t) == 1:
                t = np.array(t, dtype=np.float64)[None,:]   # [1, t]
            else:
                raise ValueError("t must be a scalar or a vector.")

            evals_t = np.power(evals, t) # [r,t]
            forecasted = np.einsum('ro,rt,nr->tno', modes, evals_t, refuns)  # [t,n,n_obs]
            if forecasted.shape[0] <= 1:
                return np.real(forecasted[0])
            else:
                return np.real(forecasted)

        except AttributeError:
            raise AttributeError("You must first fit the model and evaluate the modes with the 'self.modes' method.")

    def spectral_error(self, X = None, Y = None, left = False, axis = None):
        try:
            if X is None or Y is None:
                X = self.X
                Y = self.Y
            if left:
                error = self._lefuns(X) - self._lefuns(Y)@np.diag(self._evals.conj())
            else:
                error = self._refuns(Y) - self._refuns(X)@np.diag(self._evals)
            
            if axis is None:
                return np.linalg.norm(error, ord='fro') / np.sqrt(X.shape[0]*self._evals.shape[0])  
            else:
                return np.linalg.norm(error, axis = axis) / np.sqrt(X.shape[0]*self._evals.shape[0])
        
        except AttributeError:
            raise AttributeError("You must first fit the model and evaluate the modes with the 'self.eig' method.")

    @abstractmethod
    def eig(self):
        """
            Compute the spectral decomposition of the Koopman operator.
        """
        pass
    @abstractmethod
    def predict(self, X):
        """
            Predict the evolution of the state after a single time-step.
        """
        pass

    @abstractmethod
    def risk(self, X = None, Y = None):
        """
            Evaluate the training error (X = Y = None) or the test error.
        """
        pass
    
    def _init_kernels(self, X, Y, backend):
        self.X, self.Y = X, Y
        self.backend = parse_backend(backend, X)

        self.K_X = self.kernel(self.X, backend=self.backend)
        self.K_Y = self.kernel(self.Y, backend=self.backend)
        self.K_YX = self.kernel(self.Y, self.X, backend=self.backend)

        if self.backend == 'keops':
            self.K_X = aslinearoperator(self.K_X)
            self.K_Y = aslinearoperator(self.K_Y)
            self.K_YX = aslinearoperator(self.K_YX)

        self.dtype = self.K_X.dtype
    def _init_risk(self, X, Y):
        if (X is not None) and (Y is not None):
            K_yY = self.kernel(Y, self.Y, backend = self.backend)
            K_Xx = self.kernel(self.X, X, backend = self.backend)
            if self.backend == 'keops':
                K_yY = aslinearoperator(K_yY)
                K_Xx = aslinearoperator(K_Xx)
            _Y = Y
        else:
            K_yY = self.K_Y
            K_Xx = self.K_X
            _Y = self.Y
        r_yy = 0
        for y in _Y:
            y = y[None,:]
            r_yy += self.kernel(y,y, backend='cpu')
        r_yy = np.squeeze(r_yy)*((_Y.shape[0])**(-1))
             
        return K_yY, K_Xx, r_yy

class KernelRidgeRegression(KoopmanRegression):
    def __init__(self, kernel, tikhonov_reg = None):
        self.tikhonov_reg = tikhonov_reg
        self.kernel = kernel

    def fit(self, X, Y, backend = 'cpu'):        
        if backend != 'cpu':
            warn("Keops backend not implemented for KernelRidgeRegression. Use instead TruncatedKernelRidgeRegression. Forcing 'cpu' backend. ")
        self._init_kernels(X, Y, 'cpu')
        
    def eig(self):
        """Eigenvalue decomposition of the Koopman operator

        Returns:
            evals: Eigenvalues of the Koopman operator
            levecs: Matrix whose columns are the weigths of left eigenfunctions of the Koopman operator
            revecs: Matrix whose columns are  the weigths of right eigenfunctions of the Koopman operator
        """
        try:
            dim = self.K_X.shape[0]
            dim_inv = dim**(-1)
            sqrt_inv_dim = dim_inv**(0.5)
            K_reg = self.K_X
            if self.tikhonov_reg is not None:
                tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                K_reg += tikhonov    
            self._evals, self._levecs, self._revecs = eig(self.K_YX, K_reg, left=True, right=True)
            
            idx_ = np.argsort(np.abs(self._evals))[::-1]
            self._evals = self._evals[idx_]
            self._levecs, self._revecs = self._levecs[:,idx_], self._revecs[:,idx_]
            
            norm_r = modified_norm_sq(self._revecs,self.K_X)*dim_inv
            norm_l = modified_norm_sq(self._levecs,self.K_Y)*dim_inv

            self._revecs = self._revecs @ np.diag(norm_r**(-0.5))
            self._levecs = self._levecs @ np.diag(norm_l**(-0.5))
            self._modes_to_invert = self.K_YX@self._revecs * dim_inv

            if self.backend == 'keops':
                self._refuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.X, backend=self.backend)).matmat(self._revecs)
                self._lefuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.Y, backend=self.backend)).matmat(self._levecs)
            else:
                self._refuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.X, backend=self.backend)@self._revecs
                self._lefuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y, backend=self.backend)@self._levecs

            return self._evals, self._lefuns, self._refuns

        except AttributeError:
            raise AttributeError("You must first fit the model.")

    def predict(self, X):
        try:
            dim = self.X.shape[0]
            if self.tikhonov_reg is not None:
                if self.backend!='keops':
                    tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                    _Z = solve(self.K_X + tikhonov,self.Y, assume_a='pos')
                else:
                    _Z = IterInv(self.kernel,X,self.tikhonov_reg*dim)._matmat(self.Y)
            else:
                _Z = np.linalg.pinv(self.K_X)@self.Y
            if X.ndim == 1:
                X = X[None,:]
            _S = self.kernel(X, self.X, backend = self.backend)
            return _S@_Z
        except AttributeError:
            raise AttributeError("You must first fit the model.")
    
    def risk(self, X = None, Y = None):
        try:
            _backend = self.backend
            self.backend = 'cpu'
            K_yY, K_Xx, r = self._init_risk(X, Y)
            self.backend = _backend
            val_dim, dim = K_yY.shape
            if self.tikhonov_reg is not None:
                if self.backend !='keops':
                    tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                    C = solve(self.K_X + tikhonov, K_Xx, assume_a='pos')
                else:
                    C = IterInv(self.kernel,self.X,self.tikhonov_reg*dim)._matmat(K_Xx)
            else:
                C = np.linalg.pinv(self.K_X)@K_Xx

            r -= 2*(val_dim**(-1))*np.trace(K_yY@C)
            r += (val_dim**(-1))*np.trace(C.T@(self.K_Y@C))
            return r
        except AttributeError:
                raise AttributeError("You must first fit the model.")
class TruncatedKernelRidgeRegression(KoopmanRegression):
    def __init__(self, kernel, rank = None, tikhonov_reg = None):
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.kernel = kernel

    def fit(self, X, Y, backend = 'auto'):        
        if self.rank is None and backend != 'cpu':
            warn("Keops backend not implemented for full rank KernelRidgeRegression. Forcing 'cpu' backend.")
            self._init_kernels(X, Y, 'cpu')
        elif self.tikhonov_reg is None and backend != 'cpu':
            warn("Keops backend not implemented for unregularized KerneRegression. Forcing 'cpu' backend.")
            self._init_kernels(X, Y, 'cpu')
        else:
            self._init_kernels(X, Y, backend)
        
    def eig(self):
        """Eigenvalue decomposition of the Koopman operator

        Returns:
            evals: Eigenvalues of the Koopman operator
            levecs: Matrix whose columns are the weigths of left eigenfunctions of the Koopman operator
            revecs: Matrix whose columns are  the weigths of right eigenfunctions of the Koopman operator
        """
        try:
            dim = self.K_X.shape[0]
            dim_inv = dim**(-1)
            sqrt_inv_dim = dim_inv**(0.5)
            if self.rank is None:
                if self.tikhonov_reg is not None:
                    tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)  
                    self._evals, self._levecs, self._revecs = eig(self.K_YX, self.K_X + tikhonov, left=True, right=True)
                else:
                    self._evals, self._levecs, self._revecs = eig(self.K_YX, self.K_X, left=True, right=True)
                idx_ = np.argsort(np.abs(self._evals))[::-1]
                self._evals = self._evals[idx_]
                self._levecs, self._revecs = self._levecs[:,idx_], self._revecs[:,idx_]
            else:
                if self.tikhonov_reg is not None:
                    if self.backend == 'keops':
                        tikhonov = aslinearoperator(diags(np.ones(dim, dtype=self.dtype)*self.tikhonov_reg*dim))
                        Minv = IterInv(self.kernel, self.X, self.tikhonov_reg*dim)
                        self._evals, self._revecs = eigs(self.K_YX, self.rank, self.K_X + tikhonov,  Minv=Minv)
                        idx_ = np.argsort(np.abs(self._evals))[::-1]
                        self._evals = self._evals[idx_]
                        self._revecs = np.asfortranarray(self._revecs[:,idx_])
                        _evals, self._levecs = eigs(self.K_YX.T, self.rank, self.K_X + tikhonov,  Minv=Minv)
                        idx_ = np.argsort(np.abs(_evals))[::-1]
                        self._levecs = np.asfortranarray(self._levecs[:,idx_])
                    else:
                        tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)  
                        self._evals, self._levecs, self._revecs = eig(self.K_YX, self.K_X +tikhonov, left=True, right=True)
                        idx_ = np.argsort(np.abs(self._evals))[::-1][:self.rank]
                        self._evals = self._evals[idx_]
                        self._levecs, self._revecs = self._levecs[:,idx_], self._revecs[:,idx_]
                else:
                    self._evals, self._levecs, self._revecs = eig(np.linalg.pinv(self.K_X)@self.K_YX, left=True, right=True)
                    idx_ = np.argsort(np.abs(self._evals))[::-1][:self.rank]
                    self._evals = self._evals[idx_]
                    self._levecs, self._revecs = np.asfortranarray(self._levecs[:,idx_]), np.asfortranarray(self._revecs[:,idx_])
            

            norm_r = modified_norm_sq(self._revecs,self.K_X)*dim_inv
            norm_l = modified_norm_sq(self._levecs,self.K_Y)*dim_inv

            self._revecs = self._revecs @ np.diag(norm_r**(-0.5))
            self._levecs = self._levecs @ np.diag(norm_l**(-0.5))

            if self.backend == 'keops':
                self._refuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.X, backend=self.backend)).matmat(self._revecs)
                self._lefuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.Y, backend=self.backend)).matmat(self._levecs)
            else:
                self._refuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.X, backend=self.backend)@self._revecs
                self._lefuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y, backend=self.backend)@self._levecs

            return self._evals, self._lefuns, self._refuns

        except AttributeError:
            raise AttributeError("You must first fit the model.")

    def predict(self, X):
        try:
            dim = self.X.shape[0]
            if self.tikhonov_reg is not None:
                if self.backend!='keops':
                    tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                    _Z = solve(self.K_X + tikhonov,self.Y, assume_a='pos')
                else:
                    _Z = IterInv(self.kernel,X,self.tikhonov_reg*dim)._matmat(self.Y)
            else:
                _Z = np.linalg.pinv(self.K_X)@self.Y
            if X.ndim == 1:
                X = X[None,:]
            _S = self.kernel(X, self.X, backend = self.backend)
            return _S@_Z
        except AttributeError:
            raise AttributeError("You must first fit the model.")
    
    def risk(self, X = None, Y = None):
        try:
            _backend = self.backend
            self.backend = 'cpu'
            K_yY, K_Xx, r = self._init_risk(X, Y)
            self.backend = _backend
            val_dim, dim = K_yY.shape
            if self.tikhonov_reg is not None:
                if self.backend !='keops':
                    tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                    C = solve(self.K_X + tikhonov, K_Xx, assume_a='pos')
                else:
                    C = IterInv(self.kernel,self.X,self.tikhonov_reg*dim)._matmat(K_Xx)
            else:
                C = np.linalg.pinv(self.K_X)@K_Xx

            r -= 2*(val_dim**(-1))*np.trace(K_yY@C)
            r += (val_dim**(-1))*np.trace(C.T@(self.K_Y@C))
            return r
        except AttributeError:
                raise AttributeError("You must first fit the model.")

class LowRankKoopmanRegression(KoopmanRegression):
    def eig(self):
        """Eigenvalue decomposition of the Koopman operator

        Returns:
            evals: Eigenvalues of the Koopman operator
            levecs: Matrix whose columns are the weigths of left eigenfunctions of the Koopman operator
            revecs: Matrix whose columns are  the weigths of right eigenfunctions of the Koopman operator
        """
        try:
            dim_inv = (self.K_X.shape[0])**(-1)
            sqrt_inv_dim = dim_inv**0.5
            if self.backend == 'keops':
                C = dim_inv* self.K_YX.matmat(np.asfortranarray(self.U)) 
            else:
                C = dim_inv* self.K_YX@self.U 
            
            vals, lv, rv =  eig(self.V.T@C, left=True, right=True)
            self._evals = vals

            self._levecs = self.V@lv
            self._revecs = self.U@rv

            # sort the evals w.r.t. modulus 
            idx_ = np.argsort(np.abs(self._evals))[::-1]
            self._evals = self._evals[idx_]
            self._levecs, self._revecs = self._levecs[:,idx_], self._revecs[:,idx_]
            rv = rv[:,idx_]
            
            norm_r = modified_norm_sq(self._revecs,self.K_X)*dim_inv
            norm_l = modified_norm_sq(self._levecs,self.K_Y)*dim_inv

            self._revecs = self._revecs @ np.diag(norm_r**(-0.5))
            self._levecs = self._levecs @ np.diag(norm_l**(-0.5))

            self._modes_to_invert = rv @np.diag(self._evals*(norm_r**(-0.5)))

            if self.backend == 'keops':
                self._levecs = np.asfortranarray(self._levecs)
                self._revecs = np.asfortranarray(self._revecs)
                self._modes_to_invert = np.asfortranarray(self._modes_to_invert)
                self._refuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.X, backend=self.backend)).matmat(self._revecs)
                self._lefuns = lambda X: sqrt_inv_dim*aslinearoperator(self.kernel(X, self.Y, backend=self.backend)).matmat(self._levecs)
            else:
                self._refuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.X, backend=self.backend)@self._revecs
                self._lefuns = lambda X:  sqrt_inv_dim*self.kernel(X, self.Y, backend=self.backend)@self._levecs

            return self._evals, self._lefuns, self._refuns
        except AttributeError:
                raise AttributeError("You must first fit the model.")

    def predict(self, X):
        try:
            sqrt_dim_inv = (self.K_X.shape[0])**(-0.5)
            _Z = sqrt_dim_inv * self.V.T @ self.Y
            if X.ndim == 1:
                X = X[None,:]
            _init_K = self.kernel(X, self.X, backend = self.backend)
            if self.backend == 'keops':
                _S = sqrt_dim_inv * (aslinearoperator(_init_K).matmat(np.asfortranarray(self.U)))
            else:
                _S = sqrt_dim_inv * _init_K@self.U
            return _S@_Z 
        except AttributeError:
                raise AttributeError("You must first fit the model.")
    
    def risk(self, X = None, Y = None):
        try:
            K_yY, K_Xx, r = self._init_risk(X, Y)
            val_dim, dim = K_yY.shape
            sqrt_inv_dim = dim**(-0.5)
            V = sqrt_inv_dim*self.V
            U = sqrt_inv_dim*self.U
            if self.backend == 'keops':
                C = K_yY.matmat(np.asfortranarray(V))
                D = ((K_Xx.T).matmat(np.asfortranarray(U))).T
                E = (V.T)@self.K_Y.matmat(np.asfortranarray(V))
            else:
                C = K_yY@V
                D = (K_Xx.T@U).T
                E = (V.T)@self.K_Y@V
            r -= 2*(val_dim**(-1))*np.trace(C@D)
            r += (val_dim**(-1))*np.trace(D.T@E@D)
            return r
        except AttributeError:
                raise AttributeError("You must first fit the model.")

class ReducedRankRegression(LowRankKoopmanRegression):
    def __init__(self, kernel, rank, tikhonov_reg = None):
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.kernel = kernel
    

    def fit(self, X, Y, backend = 'auto'):
        self._init_kernels(X, Y, backend)
        dim = self.K_X.shape[0]
        inv_dim = dim**(-1)
        if self.rank is None:
            self.rank = int(dim/4)
            warn(f"Rank is not specified for ReducedRankRegression. Forcing rank={self.rank}.")

        if self.tikhonov_reg is not None:
            alpha =  self.tikhonov_reg*dim 
            K = inv_dim*(self.K_Y@self.K_X)
            if self.backend == 'keops':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=self.dtype)*alpha))
                Minv = IterInv(self.kernel, self.X, alpha)
                sigma_sq, U = eigs(K, self.rank, self.K_X + tikhonov,  Minv=Minv)                
            else:
                tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)   
                sigma_sq, U = eig(K, self.K_X + tikhonov)
        
            assert np.max(np.abs(np.imag(sigma_sq))) < 1e-10, "Numerical error in computing singular values, try to increase the regularization."
            
            sigma_sq = np.real(sigma_sq)
            sort_perm = np.argsort(sigma_sq)[::-1]
            sigma_sq = sigma_sq[sort_perm][:self.rank]
            U = U[:,sort_perm][:,:self.rank]

            #Check that the eigenvectors are real (or have a global phase at most)
            if not _check_real(U):
                raise ValueError("Computed projector is not real or a global complex phase is present. The kernel function is either severely ill conditioned or non-symmetric")

            U = np.real(U) 
            
            _nrm_sq = modified_norm_sq(U, M = KernelSquared(self.kernel,self.X, inv_dim, self.tikhonov_reg, self.backend))
            if any(_nrm_sq < _nrm_sq.max() * 4.84e-32):
                U, perm = modified_QR(U, M = KernelSquared(self.kernel,self.X, inv_dim, self.tikhonov_reg, self.backend), pivoting=True, numerical_rank=False)
                U = U[:,np.argsort(perm)]
                #self.rank = U.shape[1]
                #warn(f"Chosen rank is too high. Improving orthogonality and reducing the rank size to {self.rank}.")
            else:
                U = U@np.diag(1/_nrm_sq**(0.5))
            
            V = (self.K_X@np.asfortranarray(U))            
        else:
            if self.backend == 'keops':
                sigma_sq, V = eigsh(self.K_Y, self.rank)
                V = V@np.diag(np.sqrt(dim)/(np.linalg.norm(V,ord=2,axis=0)))
                U = lsqr(self.K_X, V)
            else:
                sigma_sq, V = eigh(self.K_Y)
                sort_perm = np.argsort(sigma_sq)[::-1]
                sigma_sq = sigma_sq[sort_perm][:self.rank]
                V = V[:,sort_perm][:,:self.rank]
                V = V@np.diag(np.sqrt(dim)/(np.linalg.norm(V,ord=2,axis=0)))
                #U = solve(self.K_X, V, assume_a='sym')
                U = lstsq(self.K_X, V)
        self.V = V 
        self.U = U

class PrincipalComponentRegression(LowRankKoopmanRegression):
    def __init__(self, kernel, rank, tikhonov_reg = None):
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.kernel = kernel

    def fit(self, X, Y, backend = 'auto'):
        self._init_kernels(X, Y, backend)
        dim = self.K_X.shape[0]
        K = self.K_X
        if self.rank is None:
            self.rank = int(dim/4)
            warn(f"Rank is not specified for PrincipalComponentRegression. Forcing rank={self.rank}.")

        if self.tikhonov_reg is not None:
            if self.backend == 'keops':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)))
            else:
                tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
            K = K + tikhonov
        if self.backend == 'keops':
            S, V = eigsh(K, self.rank)
            sigma_sq = S**2
            sort_perm = np.argsort(sigma_sq)[::-1]
            sigma_sq = sigma_sq[sort_perm]
            V = V[:,sort_perm]
            S = S[sort_perm]
        else:
            S, V = eigh(K)
            sigma_sq = S**2
            sort_perm = np.argsort(sigma_sq)[::-1]
            sigma_sq = sigma_sq[sort_perm]
            S = S[::-1][:self.rank]
            V = V[:,::-1][:,:self.rank]
        _test = S>2.2e-16
        if all(_test):            
            self.V = V * np.sqrt(dim) 
            self.U = V@np.diag(S**-1) * np.sqrt(dim)
        else:
            self.V = V[:_test] * np.sqrt(dim) 
            self.U = V[:_test]@np.diag(S[:_test]**-1) * np.sqrt(dim)
            self.rank = self.V.shape[1]
            warn(f"Chosen rank is to high! Forcing rank={self.rank}!")

class RandomizedReducedRankRegression(LowRankKoopmanRegression):
    def __init__(self, kernel, rank, tikhonov_reg = None, offset = None, powers = 2):
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.kernel = kernel
        self.offset = offset
        self.powers = powers

    def fit(self, X, Y, backend = 'auto'):
        self._init_kernels(X, Y, backend)
        dim = self.K_X.shape[0]

        if self.rank is None:
            self.rank = int(dim/4)
            warn(f"Rank is not specified for RandomizedReducedRankRegression. Forcing rank={self.rank}.")

        if self.tikhonov_reg is None:
            raise ValueError(f"Unsupported Randomized Reduced Rank Regression without Tikhonov regularization.")
        else:
            if self.backend == 'keops':
                _solve = lambda V: IterInv(self.kernel,X,self.tikhonov_reg*dim)._matmat(V)
            else:
                _solve = lambda V: solve(self.K_X + np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim),V,assume_a='pos') 

        if self.offset is None:
            self.offset = 2*self.rank

        l = self.rank + self.offset
        Omega = np.random.randn(dim,l)
        Omega = Omega @ np.diag(1/np.linalg.norm(Omega,axis=0))

        for j in range(self.powers):            
            KyO = self.K_Y@Omega
            Omega = KyO - dim*self.tikhonov_reg * _solve(KyO)
        KyO = self.K_Y@Omega

        Omega = _solve(KyO)
        Q, _ = modified_QR(Omega, M = KernelSquared(self.kernel,self.X,1/dim, self.tikhonov_reg, self.backend), pivoting=True, numerical_rank=True)        
        if Q.shape[1]<self.rank:
            print(f"Chosen rank is too high! Forcing rank to {Q.shape[1]}.")   
        C = self.K_X@Q
        sigma_sq, evecs = eigh(C.T @ (self.K_Y @ C))
        sigma_sq = sigma_sq[::-1][:self.rank]/(dim**2)
        evecs = evecs[:,::-1][:,:self.rank]
        
        U = Q @ evecs
        V = self.K_X @ U
        # error_ = np.linalg.norm(self.K_X@V/dim - (V+dim*self.tikhonov_reg*U)@np.diag(sigma_sq),ord=1)
        # if  error_> 1e-6:
        #     print(f"Attention! l1 Error in Generalized Eigenvalue Problem is {error_}")
        self.sigma_sq = sigma_sq
        self.rank = sigma_sq.shape[0]

        self.V = V
        self.U = U