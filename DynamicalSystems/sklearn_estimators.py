from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array

import numpy as np

from scipy.linalg import eig, eigh, lstsq
from scipy.sparse.linalg import aslinearoperator, eigs, eigsh, lsqr
from scipy.sparse import diags

from warnings import warn

from .sklearn_utils import sort_and_crop, weighted_norm, modified_QR, IterInv

class LowRankRegressor(RegressorMixin):
    def eig():
        pass
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
                If 'arpack', run SVD truncated to rank calling ARPACK solver functions.
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
        if self.tikhonov_reg is None:
            U, V = self._fit_unregularized(K_X, K_Y, K_YX)
        else:
            U, V = self._fit_regularized(K_X, K_Y, K_YX)     
        self.U_ = U
        self.V_ = V
        return self

    def _fit_regularized(self, K_X, K_Y):
        if self.svd_solver =='randomized':
            pass 
        else: # 'arpack' or 'full'
            dim = K_X.shape[0]
            inv_dim = dim**(-1)
            
            alpha = dim*self.tikhonov_reg
            K = inv_dim*(K_Y@K_X)
            #Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow. Prefer svd_solver == 'randomized' in such a case.
            if self.svd_solver == 'arpack':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K_X.dtype)*alpha))
                Minv = IterInv(K_X, alpha)
                sigma_sq, U = eigs(K, self.rank, K_X + tikhonov, Minv = Minv)  
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
            M = inv_dim*K_X@K_X + self.tikhonov_reg*K_X
            U_norms = weighted_norm(U, M = M)
            max_U_norm = np.max(U_norms)

            if any(U_norms < max_U_norm * 2.2e-16):  #Columns of U are too close to be linearly dependent. Perform QR factorization with pivoting to expose rank deficiency.
                U, _, columns_permutation = modified_QR(U, M = M, column_pivoting=True)
                U = U[:,np.argsort(columns_permutation)]
                if U.shape[1] < self.rank:
                    warn(f"The numerical rank of the projector is smaller than the selected rank ({self.rank}). Reducing the rank to {U.shape[1]}.")
                    self.set_params({
                        "rank": U.shape[1]
                    })
            else:
                U = U@np.diag(U_norms**-1) 
        V = K_X@np.asfortranarray(U)
        return U, V

    def _fit_unregularized(self, K_X, K_Y):
        if self.svd_solver == 'randomized':
            warn("The 'randomized' svd_solver is equivalent to 'arpack' when tikhonov_reg = None.")
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
        if self.svd_solver != 'full':
            U = np.zeros_like(V)
            for i in range(self.rank):
                U[:,i] = lsqr(K_X, V[:,i])[0] #Not optimal with this explicit loop
        else:
            U = lstsq(K_X, V)[0]
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
        if self.svd_solver not in ['full', 'arpack', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arpack\' and \'randomized\'.')
        if self.svd_solver == 'randomized' and self.iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if self.svd_solver == 'randomized' and self.n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        if self.svd_solver == 'full' and self.backend == 'keops':
            raise ValueError('Invalid backend and svd_solver combination. \'keops\' backend is not compatible with \'full\' svd_solver.')
        return