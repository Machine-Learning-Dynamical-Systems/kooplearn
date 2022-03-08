from scipy.sparse.linalg import eigs, aslinearoperator
from scipy.sparse import identity
import numpy as np

def reduced_rank_regression(data, evolved_data, kernel, rank, regularizer=1e-8, backend='torch'):
    
    data_kernel = kernel(data, backend=backend)
    evolved_data_kernel = kernel(evolved_data, backend=backend)
    cross_kernel = kernel(evolved_data, data, backend=backend)
    if backend == 'torch':
        data_kernel = data_kernel.cpu().numpy()
        evolved_data_kernel = evolved_data_kernel.cpu().numpy()
        cross_kernel = cross_kernel.cpu().numpy()

    V_r = _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer, backend)
    dim = data.shape[0]
    U = aslinearoperator((V_r.T)@aslinearoperator(cross_kernel))
    V = aslinearoperator(aslinearoperator(evolved_data_kernel/dim)@V_r)
    B = aslinearoperator(data_kernel) + aslinearoperator(identity(dim, dtype=U.dtype)*(regularizer*dim))
    return eigs(V@U, rank, B)

def _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer, backend):
    dim = data_kernel.shape[0]
    _rescaled_evolved_data_kernel = evolved_data_kernel/dim
    if backend=='torch':
        A = aslinearoperator(data_kernel@evolved_data_kernel)
    else:
        A = aslinearoperator(data_kernel)@aslinearoperator(_rescaled_evolved_data_kernel)
    B = aslinearoperator(data_kernel) + aslinearoperator(identity(dim, dtype=A.dtype)*(regularizer*dim))
    _, V = eigs(A, rank, B)
    return aslinearoperator(modified_Gram_Schmidt(V, _rescaled_evolved_data_kernel))

def modified_Gram_Schmidt(A, M=None):
    dim = A.shape[0]
    vecs = A.shape[1]
    if M is None:
        M = identity(dim)
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, vecs):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], M@q)
            q = q - rij*Q[:,i]
        rjj = np.sqrt(np.vdot(q, M@q))
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q