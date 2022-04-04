import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import LinearOperator
from pykeops.numpy import Vi
import matplotlib.pyplot as plt

__useTeX__ = True
if __useTeX__:
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
        #"font.family": "sans-serif",
        #"font.sans-serif": ["Computer Modern Serif"]
    })

def plot_eigs(eigs, log = False, labels = None):
    if isinstance(eigs, np.ndarray):
        if eigs.ndim == 1:
            eigs = eigs[:, np.newaxis]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for idx, eig in enumerate(eigs):
        kwargs = {
            'c': colors[idx],
            's': 2
        }
        if labels is not None:
            kwargs['label'] = labels[idx]
        if log:
            ax.scatter(np.log(np.abs(eig)), np.angle(eig), **kwargs)
            ax.set_xlabel(r'$\log(|\lambda_i|)$')
            ax.set_ylabel(r'$\angle\lambda_i$')
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        else:
            ax.scatter(eig.real, eig.imag, **kwargs)
            ax.set_xlabel('Real part')
            ax.set_ylabel('Imaginary part')
            ax.set_aspect('equal')
    if labels is not None:
        ax.legend(frameon=False)

    unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="k",
            fill=False,
            label="Unit circle",
            linestyle="-",
        )
    if not log:
        ax.add_artist(unit_circle)
    return fig, ax

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