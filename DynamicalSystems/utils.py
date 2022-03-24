import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import LinearOperator
from pykeops.numpy import Vi
import matplotlib.pyplot as plt

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
        log = False,
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

        (points,) = ax.plot(
            eigs.real, eigs.imag, "r+", label="Eigenvalues"
        )
        lim = 1.1
        supx, infx, supy, infy = lim, -lim, lim, -lim
        
        # set limits for axis
        ax.set_xlim((infx, supx))
        ax.set_ylim((infy, supy))
        

        plt.ylabel("Imaginary part")
        plt.xlabel("Real part")
        
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="k",
            fill=False,
            label="Unit circle",
            linestyle="-",
        )
        ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle("--")
        ax.grid(True)

        ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best", frameon=False))
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

def modified_QR(A, backend, M=None):
    backend = parse_backend(backend, A)
    dim = A.shape[0]
    vecs = A.shape[1]
    if M is None:
        if backend == 'keops':
            M = identity(dim, dtype= A.dtype)
        else:
            M = np.eye(dim, dtype = A.dtype)
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, vecs):
        q = np.asfortranarray(A[:,j])
        for i in range(0, j):
            rij = np.vdot(Q[:,i], M@q)
            q = q - rij*Q[:,i]
        rjj = np.sqrt(np.vdot(q, M@q))
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q

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