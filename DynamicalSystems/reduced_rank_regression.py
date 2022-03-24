
from cmath import sqrt
from scipy.sparse.linalg import eigs, aslinearoperator, LinearOperator
import scipy.linalg
from scipy.sparse import diags
from DynamicalSystems.utils import modified_QR, parse_backend, _check_real, IterInv
import numpy as np
from warnings import warn

class KoopmanRegression:
    def __init__(self, data, evolved_data, kernel, rank, regularizer, center_kernel=False, backend='auto'):
        self.backend = parse_backend(backend, data)
        self.kernel = kernel
        self.X, self.Y = data, evolved_data
        self.rank = rank
        self.reg = regularizer
        self.center_kernel = center_kernel
        
        if self.backend == 'keops':
            self.K_X = aslinearoperator(self.kernel(self.X, backend=self.backend))
            self.K_Y = aslinearoperator(self.kernel(self.Y, backend=self.backend))
            self.K_YX = aslinearoperator(self.kernel(self.Y, self.X, backend=self.backend))
        else:
            self.K_X = self.kernel(self.X, backend=self.backend)
            self.K_Y = self.kernel(self.Y, backend=self.backend)
            self.K_YX = self.kernel(self.Y, self.X, backend=self.backend)
        
        if center_kernel:
            dK_Y = _center_kernel(self.kernel, self.Y, self.Y, self.X, averaged_indices=(True, True), backend=self.backend)
            dK_YX = _center_kernel(self.kernel, self.Y, self.X, self.X, averaged_indices=(True, False), backend=self.backend)
            if self.backend == 'keops':
                self.K_Y += dK_Y
                self.K_YX += dK_YX
            else:
                Id = np.eye(self.X.shape[0], dtype = self.X.dtype)
                self.K_Y += dK_Y.matmat(Id)
                self.K_YX += dK_YX.matmat(Id)
        print("INIT: Computing low-rank-projection")
        self.V, self.U = self._low_rank_projector()
    
    def _low_rank_projector(self):
        #For the moment data_kernel = LinearOperator, evolved_data_kernel = LinearOperator
        dim = self.K_X.shape[0]
        inverse_dim = dim**-1
        
        K = inverse_dim*(self.K_Y@self.K_X)  
        K.dtype = self.K_X.dtype
        
        if self.backend == 'keops':
            tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K.dtype)*(self.reg*dim)))
            alpha =  self.reg*dim
            Minv = IterInv(self.kernel, self.X, alpha)
            sigma_sq, U = eigs(K, self.rank, self.K_X + tikhonov,  Minv=Minv)
        else:
            tikhonov = np.eye(dim, dtype=K.dtype)*(self.reg*dim)
            sigma_sq, U = scipy.linalg.eig(K, self.K_X + tikhonov)
    
            assert np.max(np.abs(np.imag(sigma_sq))) < 1e-8
            sigma_sq = np.real(sigma_sq)
            sort_perm = np.argsort(sigma_sq)[::-1]
            sigma_sq = sigma_sq[sort_perm][:self.rank]
            U = U[:,sort_perm][:,:self.rank]
     
        #Check that the eigenvectors are real (or have a global phase at most)
        if not _check_real(U):
            U_global_phase_norm = np.angle(U).std()
            if U_global_phase_norm  > 1e-8:
                raise ValueError("Computed projector is not real. The kernel function is either severely ill conditioned or non-symmetric")
            else:
                #It has a global complex phase, take absolute.
                U = np.abs(U)
        else:
            U = np.real(U)
        
        U = modified_QR(U, self.backend, inverse_dim*(self.K_X@(self.K_X + tikhonov)))
        V = inverse_dim*(self.K_X@np.asfortranarray(U))
        sigma = np.sqrt(np.real(sigma_sq))
        return V*(sigma**-1), U*sigma
    
    def eig(self, X=None):
        if self.backend == 'keops':
            C = self.K_YX.matmat(np.asfortranarray(self.U))
        else:
            C = self.K_YX@self.U
        vals, lv, rv =  scipy.linalg.eig(self.V.T@C, left=True, right=True)
        if X is not None:
            Kr = self.kernel(X, self.X, backend=self.backend)
            Kl = self.kernel(X, self.X, backend=self.backend)
            if self.center_kernel:
                warn("The left eigenfunctions are evaluated with the standard kernel, i.e. without centering.")
            if self.backend == 'keops':
                return vals, aslinearoperator(Kl).matmat(lv), aslinearoperator(Kr).matmat(rv) 
            else:
                return vals, Kl@lv, Kr@rv
        else:
            return vals, self.V@lv, self.U@rv
    def modes(self, f = None):
        if f is not None:
            observable = f(self.X)
        else:
            observable = self.X
        self.U.T@observable

def _center_kernel(kernel, X, Y, D, averaged_indices, backend):
    K_Y = kernel(Y, D, backend=backend).sum(1).squeeze() #Vector
    K_X = kernel(X, D, backend=backend).sum(1).squeeze() #Vector
    K_D = kernel(D, D, backend=backend).sum(0).squeeze().sum(0) #Scalar
    scale = np.array(K_X.shape[0]**-1).astype(K_X.dtype)
    if averaged_indices == (False, False):
        warn("No averaging correction computed. Returning None")
        return None
    if averaged_indices == (True, False):
        def _matvec(w):
            w = w.squeeze()
            return np.full_like(w, -np.dot(w, K_Y)*scale)
    elif averaged_indices == (False, True):
        def _matvec(w):
            w = w.squeeze()
            W = w.sum()
            return -K_X*W*scale
    else:
        def _matvec(w):
            w = w.squeeze()
            W = w.sum()
            #Default choice average both indices
            return (-K_X*W - np.dot(w, K_Y) + W*K_D*scale)*scale
    return LinearOperator((X.shape[0], X.shape[0]), matvec =  _matvec, dtype= K_X.dtype)