from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh
from scipy.sparse import diags
from .utils import parse_backend, IterInv, _check_real, modified_QR
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
            sqrt_dim = (self.K_X.shape[0])**(0.5)
            evaluated_observable = observable(self.X)
            if evaluated_observable.ndim == 1:
                evaluated_observable = evaluated_observable[:,None]
            U_tilde = np.array(self._revecs)
            if self.backend == 'keops':
                F = U_tilde.H.matmat(np.asfortranarray(evaluated_observable))
                D = (U_tilde.H)@self.K_X.matmat(np.asfortranarray(self._revecs))
            else:
                F = U_tilde.H@evaluated_observable
                D = (U_tilde.H)@self.K_X@U_tilde
            
            self._modes = sqrt_dim*solve(D, F, assume_a='her')  
            self._modes_observable = observable  
            return self._modes
        except AttributeError:
            try:
                self.eig()
                return self.modes(observable)
            except AttributeError:
                raise AttributeError("You must first fit the model.")
    
    def forecast(self, X, t, which = None):
        try:
            evaluated_observable = self._modes_observable(X)
            if evaluated_observable.ndim == 1:
                evaluated_observable = evaluated_observable[:,None]

            if which is not None:
                evals = self._evals[which]
                revecs = self._revecs[:,which]
                modes = self._modes[which,:]
            else:
                evals = self._evals
                revecs = self._revecs
                modes = self._modes

            sqrt_inv_dim = (self.K_X.shape[0])**(-0.5)

            if self.backend == 'keops':
                D = aslinearoperator(self.kernel(X, self.X, backend=self.backend)).matmat(np.asfortranarray(revecs))
            else:
                D = self.kernel(X, self.X, backend=self.backend)@revecs
            evolved_evals = (evals**t)[:,None]
            return sqrt_inv_dim*np.real(evolved_evals*(D@modes))

        except AttributeError:
            raise AttributeError("You must first fit the model and evaluate the modes with the 'self.modes' method.")

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
            warn("Keops backend not implemented for KernelRidgeRegression. Forcing 'cpu' backend.")
        self._init_kernels(X, Y, 'cpu')
        
    def eig(self):
        """Eigenvalue decomposition of the Koopman operator

        Returns:
            eigvals: Eigenvalues of the Koopman operator
            left_eigvecs: Matrix whose columns are the left eigenvectors of the Koopman operator
            right_eigvecs: Matrix whose columns are the right eigenvectors of the Koopman operator
        """
        
        dim = self.K_X.shape[0]
        if self.tikhonov_reg is not None:
            tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
            C = solve(self.K_X+tikhonov, self.K_YX, assume_a='pos') 
        else:
            C = np.linalg.pinv(self.K_X)@ self.K_YX
        
        self._evals, self._levecs, self._revecs = eig(C, left=True, right=True)

        _rank = self._evals.shape[0]
        if self.backend == 'keops':
            norm_r = np.diag(_rank*self._revecs.T@(self.K_X.matmat(np.asfortranarray(self._revecs))))
            norm_l = np.diag(_rank*self._levecs.T@(self.K_Y.matmat(np.asfortranarray(self._levecs))))
        else:
            norm_r = np.diag(_rank*self._revecs.T@self.K_X@self._revecs)
            norm_l = np.diag(_rank*self._levecs.T@self.K_Y@self._levecs)

        self._revecs = self._revecs @ np.diag(norm_r**(-0.5))
        self._levecs = self._levecs @ np.diag(norm_l**(-0.5))

        return self._evals, self._levecs, self._revecs

    def predict(self, X):
        try:
            if self.tikhonov_reg is not None:
                tikhonov = np.eye(self.n_samples, dtype=self.dtype)*(self.tikhonov_reg*self.n_samples)
                _Z = solve(self.K_X + tikhonov,self.Y, assume_a='pos') 
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
            K_yY, K_Xx, r = self._init_risk(X, Y)
            val_dim, dim = K_yY.shape
            if self.tikhonov_reg is not None:
                tikhonov = np.eye(dim, dtype=self.dtype)*(self.tikhonov_reg*dim)
                C = solve(self.K_X + tikhonov, K_Xx, assume_a='pos')
            else:
                C = np.linalg.pinv(self.K_X)@K_Xx

            r -= 2*(val_dim**(-1))*np.trace(K_yY@C)
            r += (val_dim**(-1))*np.trace(C.T@self.K_Y@C)
            return r
        except AttributeError:
                raise AttributeError("You must first fit the model.")

class LowRankKoopmanRegression(KoopmanRegression):
    def eig(self):
        """Eigenvalue decomposition of the Koopman operator

        Returns:
            eigvals: Eigenvalues of the Koopman operator
            left_eigvecs: Matrix whose columns are the left eigenvectors of the Koopman operator
            right_eigvecs: Matrix whose columns are the right eigenvectors of the Koopman operator
        """
        try:
            dim_inv = (self.K_X.shape[0])**(-1)
            if self.backend == 'keops':
                C = dim_inv* self.K_YX.matmat(np.asfortranarray(self.U)) 
            else:
                C = dim_inv* self.K_YX@self.U 
            
            vals, lv, rv =  eig(self.V.T@C, left=True, right=True)
            self._evals = vals 
            self._levecs = self.V@lv
            self._revecs = self.U@rv

            _rank = self._evals.shape[0]
            if self.backend == 'keops':
                norm_r = np.diag(_rank*self._revecs.T@(self.K_X.matmat(np.asfortranarray(self._revecs))))
                norm_l = np.diag(_rank*self._levecs.T@(self.K_Y.matmat(np.asfortranarray(self._levecs))))
            else:
                norm_r = np.diag(_rank*self._revecs.T@self.K_X@self._revecs)
                norm_l = np.diag(_rank*self._levecs.T@self.K_Y@self._levecs)

            self._revecs = self._revecs @ np.diag(norm_r**(-0.5))
            self._levecs = self._levecs @ np.diag(norm_l**(-0.5))

            return self._evals, self._levecs, self._revecs
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
            M = inv_dim*(self.K_X@(self.K_X + tikhonov)) 
            U = modified_QR(U, self.backend, M = M)
            V = (self.K_X@np.asfortranarray(U))            
        else:
            if self.backend == 'keops':
                sigma_sq, V = eigsh(self.K_Y, self.rank)
                V = V@np.diag(np.sqrt(dim)/(np.linalg.norm(V,ord=2,axis=0)))
                M = IterInv(self.kernel, self.X)
                U = np.empty_like(V)
                for i in range(self.rank):
                    U[:,i] = M._matvec(V[:,i])
            else:
                sigma_sq, V = eigh(self.K_Y)
                sort_perm = np.argsort(sigma_sq)[::-1]
                sigma_sq = sigma_sq[sort_perm][:self.rank]
                V = V[:,sort_perm][:,:self.rank]
                V = V@np.diag(np.sqrt(dim)/(np.linalg.norm(V,ord=2,axis=0)))
                U = solve(self.K_X, V, assume_a='sym')
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

        self.V = V * np.sqrt(dim) 
        self.U = V@np.diag(S**-1) * np.sqrt(dim)

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
        if self.backend == 'keops':
            raise ValueError(f"Unsupported backend '{self.backend}' for Randomized Reduced Rank Regression.")

        if self.tikhonov_reg is None:
            raise ValueError(f"Unsupported unregularized Randomized Reduced Rank Regression.")

        if self.rank is None:
            self.rank = int(np.trace(self.K_Y)/np.linalg.norm(self.K_Y,ord =2))+1
            print(f'Numerical rank of the output kernel is approximatly {int(self.rank)} which is used.')

        if self.offset is None:
            self.offset = 2*self.rank

        l = self.rank + self.offset
        Omega = np.random.randn(dim,l)
        Omega = Omega @ np.diag(1/np.linalg.norm(Omega,axis=0))
        for j in range(self.powers):
            KyO = self.K_Y@Omega
            Omega = KyO - dim*self.tikhonov_reg*solve(self.K_X+dim*self.tikhonov_reg*np.eye(dim),KyO,assume_a='pos')
        KyO = self.K_Y@Omega
        Omega = solve(self.K_X+dim*self.tikhonov_reg*np.eye(dim), KyO, assume_a='pos')
        Q = modified_QR(Omega, self.backend, M = self.K_X@self.K_X/dim+self.K_X*self.tikhonov_reg)
        if Q.shape[1]<self.rank:
            print(f"Actual rank is smaller! Detected rank is {Q.shape[1]} and is used.")   
        C = self.K_X@Q
        sigma_sq, evecs = eigh((C.T @ self.K_Y) @ C)
        sigma_sq = sigma_sq[::-1][:self.rank]/(dim**2)
        evecs = evecs[:,::-1][:,:self.rank]
        
        U = Q @ evecs
        V = self.K_X @ U
        error_ = np.linalg.norm(self.K_X@V/dim - (V+dim*self.tikhonov_reg*U)@np.diag(sigma_sq),ord=1)
        if  error_> 1e-6:
            print(f"Attention! l1 Error in Generalized Eigenvalue Problem is {error_}")
        self.sigma_sq = sigma_sq
        self.rank = sigma_sq.shape[0]

        self.V = V
        self.U = U