from logging import warning
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg, aslinearoperator, LinearOperator
from scipy.linalg import eigh, solve, solve_triangular
from pykeops.numpy import Vi
import matplotlib.pyplot as plt
from warnings import warn

__useTeX__ = True
if __useTeX__:
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
        #"font.family": "sans-serif",
        #"font.sans-serif": ["Computer Modern Serif"]
    })

def plot_eigs(
        eigs,
        log = None,
        figsize=(8, 8),
        title="",
        dpi=None,
        filename=None,
        ax = None,
        style = 'r+',
        label = 'Eigenvalues'
    ):
        _given_axis = True
        if ax is None:
            ## Adapted from package pyDMD
            if dpi is not None:
                plt.figure(figsize=figsize, dpi=dpi)
            else:
                plt.figure(figsize=figsize)

            plt.title(title)
            plt.gcf()
            ax = plt.gca()
            _given_axis = False

        if log is None:
            ax.plot(
                eigs.real, eigs.imag, style, label=label
            )
            lim = 1.1
            supx, infx, supy, infy = lim, -lim, lim, -lim

            # set limits for axis
            ax.set_xlim((infx, supx))
            ax.set_ylim((infy, supy))

        else:
            ax.plot(
            np.log(np.abs(eigs)), np.angle(eigs), style, label=label
            )

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")
        
        if not _given_axis:
            if log is None:
                unit_circle = plt.Circle(
                    (0.0, 0.0),
                    1.0,
                    color="k",
                    fill=False,
                    linestyle="-",
                )
                ax.add_artist(unit_circle)
            else:
                line_ = plt.Line2D(
                    (0.0, 0.0),
                    (-np.pi,np.pi),
                    color="k",
                    linestyle="-",
                )
                ax.add_artist(line_)
            # Dashed grid
            gridlines = ax.get_xgridlines() + ax.get_ygridlines()
            for line in gridlines:
                line.set_linestyle("--")
            ax.grid(True)
            if log is None:
                ax.set_aspect("equal")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        return ax

def parse_backend(backend, X):
        if backend == 'keops':
            return backend
        elif backend == 'cpu':
            return backend
        elif backend == 'auto':
            if X.shape[0] < 2000:
                return 'cpu'
            else:
                return 'keops'
        else:
            raise ValueError(f"Unrecognized backend '{backend}'. Accepted values are 'auto', 'cpu' or 'keops'.")
def _is_real(V, eps = 1e-8):
    if np.max(np.abs(np.imag(V))) > eps:
        return False
    else:
        return True

class IterInv(LinearOperator):
    """
    Adapted from scipy
    IterInv:
       helper class to repeatedly solve M*x=b
       using an iterative method.
    """
    def __init__(self, kernel, X, alpha, eps=1e-6):
        self.M = kernel(X, backend='keops')
        self.dtype = X.dtype
        self.shape = self.M.shape
        self.alpha = alpha
        self.eps = eps

    def _matvec(self, x):
        _x = Vi(x[:, np.newaxis])
        b = self.M.solve(_x, alpha=self.alpha, eps = self.eps)
        return b

    def _matmat(self, x):
        _x = Vi(x)
        b = self.M.solve(_x, alpha=self.alpha, eps = self.eps)
        return b

class KernelSquared(LinearOperator):
    """
    Adapted from scipy
    KernelSquared:
       helper class to repeatedly apply alpha*K@K+beta*K.
    """
    def __init__(self, kernel, X, alpha, beta, backend):
        k = kernel(X, backend=backend)
        if backend == 'keops':
            self.M = aslinearoperator(k)
        else:
            self.M = k
        self.dtype = X.dtype
        self.shape = self.M.shape
        self.alpha = alpha
        self.beta = beta

    def _matvec(self, x):
        v = np.ascontiguousarray(self.M @ x)
        return self.alpha * self.M @ v + self.beta * v

def modified_norm_sq(A, M=None):
    dtype = A.dtype
    if dtype=='complex':
        herm = lambda X: np.conj(X.T)  
    else:
        herm =  lambda X: X.T
    
    if len(A.shape)==1:
        A = A[:,None]
    dim, vecs = A.shape

    _nrm = np.empty(vecs, dtype=dtype)
    for k in range(vecs):
        if M is None:
            _nrm[k] = herm(A[:,k]) @ np.ascontiguousarray(A[:,k])
        else:
            _nrm[k] = herm(A[:,k]) @ (M@np.ascontiguousarray(A[:, k]))

    _nrm = np.real(_nrm)
    return _nrm #if A.shape[1]>1 else _nrm[0]

def weighted_norm(A, M=None):
    if len(A.shape)==1:
        A = A[:,None] #A.shape = [dim, vecs]
    A_conj = np.conj(A.copy())
    if M is not None::
        if isinstance(M, LinearOperator):
            A = M.matmat(np.asfortranarray(A))  
        else:
            A = M@A
    norm = np.sum(A_conj*A, axis=0)
    return np.real(norm)

def modified_QR(A, M = None, pivoting = False, numerical_rank = False, return_R = False):
    """
    Applies the row-wise Gram-Schmidt method to A
    and returns Q with M-orthonormal columns for M symmetric positive definite. 
    
    Parameters:
    
    return_R = True 
        additionally returns R such that A = Q R.
    pivoting=True 
        uses column-wise pivoting and returns permuation perm such that A[:,perm] = Q R
    numerical_rank=True 
        detects when numerical rank of A is reached and stops so that rank = Q.shape[1] and A[:,perm[:rank]] = Q R[:,:rank] 
        while A[:,perm[rank:]] is in the kenrel of M 

    For M being numerically rank deficient, it is recomended to use pivoting=True and numerical_rank=True.

    """
    dim, vecs = A.shape
    rank = vecs
    dtype = A.dtype
    if dtype=='complex':
        herm = lambda X: np.conj(X.T)  
    else:
        herm =  lambda X: X.T

    Q = np.copy(A)
    R = np.zeros((vecs,vecs), dtype=dtype)
    _perm = np.arange(0,vecs)

    if (numerical_rank) and (not pivoting):
        pivoting = True
        warn(f"Forcing pivoting to detect numerical rank.")

    if pivoting:
        eps, tau = 1e-8, 1e-2
        _nrm = weighted_norm(A, M=M)
        _eps = eps * _nrm
        _nrm_max = _nrm.max()
    else:
        _nrm_max = 1.

    for k in range(0, vecs):
        if pivoting:
            idx = np.argmax(_nrm[k:])
            _in = [k, k + idx]
            _swap = [k + idx,k]
            #Column pivoting
            _perm[ _in] = _perm[_swap] 
            Q[:, _in] = Q[:,_swap]
            R[:k, _in] = R[:k,_swap]
            _nrm[ _in]= _nrm[_swap]
        
        if k > 0:
            Q_k = np.ascontiguousarray(Q[:, k].copy())
            Q_pre_k = Q[:,:k]
            if M is not None:
                if isinstance(M, LinearOperator):
                    Q_k = M.matvec(Q_k)  
                else:
                    Q_k = np.dot(M, Q_k)
            
            Q_norm = Q_pre_k.T@Q_k

            R[:k,k] += Q_norm
            Q[:,k] -= np.dot(Q_pre_k,Q_norm)
  
        _nrm_k = weighted_norm(Q[:,k], M=M)
        if pivoting and numerical_rank and (_nrm_k < _nrm_max * 4.84e-32):
            rank = k 
            break

        R[k, k] = _nrm_k**(0.5) 
        Q[:, k] = Q[:, k] / R[k, k]
        if k < vecs - 1:
            Q_k = np.ascontiguousarray(Q[:, k].copy())
            if M is not None:
                if isinstance(M, LinearOperator):
                    Q_k = M.matvec(Q_k)  
                else:
                    Q_k = np.dot(M, Q_k)
            R[k,k+1:] = np.sum(np.conj(Q_k)[:, None]*Q[:, k+1:], axis = 0)        
            Q[:,k+1:] -= np.outer(Q_k, R[k,k+1:])
            if pivoting:
                _nrm[k+1:] -= np.abs(R[k,k+1:])**2
                _test = _nrm[k+1:] < _eps[k+1:] / tau
                if any(_test):
                    _nrm[k+1:][_test] = weighted_norm(Q[:,k+1:][:,_test], M=M)
                    _eps[k+1:][_test] = eps*_nrm[k+1:][_test]           
    if return_R:        
        if pivoting:
            return Q[:,:rank], R[:rank,:], _perm 
        else:
            return Q[:,:rank], R[:rank,:]
    else:
        if pivoting:
            return Q[:,:rank], _perm 
        else:
            return Q[:,:rank] 
            
def rSVD(Kx,Ky,reg, rank= None, powers = 2, offset = 5, tol = 1e-6):
    n = Kx.shape[0]

    if rank is None:
        rank = int(np.trace(Ky)/np.linalg.norm(Ky,ord =2))
        print(f'Numerical rank of the output kernel is approximatly {rank}')

    l = rank+offset
    Omega = np.random.randn(n,l)
    Omega = Omega @ np.diag(1/np.linalg.norm(Omega,axis=0))
    for j in range(powers):
        KyO = Ky@Omega
        Omega = KyO - n*reg*solve(Kx+n*reg*np.eye(n),KyO,assume_a='pos')
    KyO = Ky@Omega
    Omega = solve(Kx+n*reg*np.eye(n), KyO, assume_a='pos')
    Q = modified_QR(Omega, backend = 'cpu', M = Kx@Kx/n+Kx*reg)
    if Q.shape[1]<rank:
        print(f"Actual rank is smaller! Detected rank is {Q.shape[1]}")   
    C = Kx@Q
    svals2, evecs = eigh((C.T @ Ky) @ C)
    svals2_ = svals2[::-1]/(n**2)
    svals2 = svals2_[:rank]
    evecs = evecs[:,::-1][:,:rank]
    
    print(svals2)
    U = Q @ evecs
    V = Kx @ U
    error_ = np.linalg.norm(Ky@V/n - (V+n*reg*U)@np.diag(svals2),ord=1)
    if  error_> 1e-6:
        print(f"Attention! l1 Error in GEP is {error_}")
        #num_rank = np.sum(svals2_ / np.sqrt(svals2_*svals2_)>1e-16)
        #print(f'Numerical rank of the estimator is approximatly {num_rank}')

    return U, V, svals2

