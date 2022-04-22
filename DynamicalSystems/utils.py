from logging import warning
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy.linalg import eigh, solve
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
    ):
        ## Adapted from package pyDMD
        if dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure(figsize=figsize)

        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        if log is None:
            (points,) = ax.plot(
                eigs.real, eigs.imag, "r+", label="Eigenvalues"
            )
            lim = 1.1
            supx, infx, supy, infy = lim, -lim, lim, -lim

            # set limits for axis
            ax.set_xlim((infx, supx))
            ax.set_ylim((infy, supy))

        else:
            (points,) = ax.plot(
            np.log(np.abs(eigs)), np.angle(eigs), "r+", label="Eigenvalues"
            )

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")
        
        if log is None:
            unit_circle = plt.Circle(
                (0.0, 0.0),
                1.0,
                color="k",
                fill=False,
                label="Unit circle",
                linestyle="-",
            )
            ax.add_artist(unit_circle)
        else:
            line_ = plt.Line2D(
                (0.0, 0.0),
                (-np.pi,np.pi),
                color="k",
                label="Imaginary axis",
                linestyle="-",
            )
            ax.add_artist(line_)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("--")
        ax.grid(True)

        ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best", frameon=False))
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

def _check_real(V, eps = 1e-8):
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

def modified_QR(A, backend, M=None):
    """
    Applies the Gram-Schmidt method to A
    and returns Q with M-orthonormal columns
    """
    dim, vecs = A.shape
    numerical_rank = vecs
    if M is None:
        if backend == 'keops':
            M = identity(dim, dtype= A.dtype)
        else:
            M = np.eye(dim, dtype = A.dtype)
    Q = np.copy(A)
    R = np.zeros((vecs,vecs), dtype=A.dtype)
    _perm = np.arange(0,vecs) 
    for k in range(0, vecs):
        if backend == 'keops':
            _nrm = np.diag(Q[:,k:].T @ M.matmat(np.asfortranarray(Q[:, k:])))
        else:
            _nrm = np.diag(Q[:,k:].T @ M @Q[:, k:])
        idx = np.argmax(_nrm)
        idx = 0
        _perm[[k,k+idx]] = _perm[[k+idx,k]] 
        Q[:,[k,k+idx]] = Q[:,[k+idx,k]]
        R[:,[k,k+idx]] = R[:,[k+idx,k]]
        R[k, k] = np.sqrt(np.abs(_nrm[idx]))  
        if not R[k, k] <1e-12:
            if 0:#k>0:
                if backend == 'keops':
                    tmp = Q[:,:k].T@(M.matvec(Q[:,k]))
                else:
                    tmp = Q[:,:k].T@(M@Q[:,k])
                #tmp = Q[:,:k].T@(Q[:,k])
                R[:k,k] += tmp
                Q[:,k] -= Q[:,:k]@tmp
                
            Q[:, k] = Q[:, k]/R[k, k]
            if k<vecs-1:
                if backend == 'keops':
                    R[k,k+1:] = (M.matvec(Q[:,k])).T @  Q[:,k+1:]
                else:
                    R[k,k+1:] = (M@Q[:,k]).T @  Q[:,k+1:]
                Q[:,k+1:] -= np.outer(Q[:,k], R[k,k+1:])
        else:
            numerical_rank = k 
            break
    #print('orthogonality error:', np.linalg.norm(R-np.diag(np.diag(R)), ord = 1))        
    return Q[:,_perm][:,:numerical_rank]#, R[:,perm_][:rank,:]

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

