from multiprocessing.sharedctypes import Value
from attr import evolve
from scipy.sparse.linalg import eigs, eigsh, aslinearoperator, LinearOperator, cg
from scipy.linalg import eig
from scipy.sparse import diags, identity
from pykeops.numpy import Vj
import numpy as np
from warnings import warn
import tqdm
    
def reduced_rank_regression(data, evolved_data, kernel, rank, regularizer=None, center_kernel = False):
    # Defining kernels
    data_kernel = aslinearoperator(kernel(data, data))
    evolved_data_kernel = aslinearoperator(kernel(evolved_data, evolved_data))
    #Working with the transpose of the matrix in the notes for easier implementation
    cross_kernel = aslinearoperator(kernel(data, evolved_data))
  
    #Correcting Kernel if needed
    if center_kernel:
        evolved_data_kernel += _center_kernel(kernel, evolved_data, evolved_data, data, averaged_indices=(True, True))
        cross_kernel += _center_kernel(kernel, data, evolved_data, data, averaged_indices=(False, True))
    
    print("Low rank projection")
    proj = np.asfortranarray(_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer))
    dim = proj.shape[0]
    #Neither U nor V are normalised by 1/dim because from _get_low_rank_projector we spare a factor dim
    U = cross_kernel.matmat(proj)
    U = U.T #[rank, dim]
    if regularizer is None:
        T = proj
    else:        
        T = np.asfortranarray(evolved_data_kernel.matmat(proj)) # [dim, rank]
    print("Final diagonalization")
    
    # U@K^-1@T w = λw
    # C := K_reg^-1@V
    if regularizer is None:
        alpha = 1e-10 #Very small reg parameter to solve least squares.
    else:
        alpha = regularizer*dim
    
    K = aslinearoperator(diags(alpha * np.ones(dim, dtype=data_kernel.dtype), dtype=data_kernel.dtype)) + data_kernel
    C = np.empty_like(proj)
    for i in tqdm.tqdm(range(rank), desc = "Solving CG iteration"):
        x, info = cg(K, T[:,i], tol=1e-10)
        if info > 0:
            warn("Convergence not achieved")
            C[:,i] = 0
        elif info < 0:
            raise ValueError("Illegal input or breakdown")
        C[:, i] = x
    #C = kernel(data).solve(Vj(T), alpha=alpha, sum_scheme='kahan_scheme')
    vals, vecs =  eig(U@C)
    return vals, C@vecs

def _low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer):
    #For the moment data_kernel = LinearOperator, evolved_data_kernel = LinearOperator
    dim = data_kernel.shape[0]
    inverse_dim = dim**-1

    if regularizer is None:
        #Momentary solution
        _, V = eigsh(evolved_data_kernel, rank)
    else:
        K = inverse_dim*((data_kernel.T)@evolved_data_kernel)     
        tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K.dtype)*(regularizer*dim)))
        _, V = eigs(K, rank, data_kernel + tikhonov)
        
        #Check that the eigenvectors are real (or have a global phase at most)
        if not _check_real(V):
            V_global_phase_norm = np.angle(V).std()
            if V_global_phase_norm  > 1e-8:
                raise ValueError("Computed projector is not real. The kernel function is either severely ill conditioned or non-symmetric")
            else:
                #It has a global complex phase, take absolute.
                V = np.abs(V)
        else:
            V = np.real(V)
            V = modified_QR(V, evolved_data_kernel)
        #V = modified_QR(V, evolved_data_kernel)

    #V now are orthonormal w.r.t. K_P and not K_P/dim. They should be normalised by multiplying by sqrt(dim). Let's skip this and lose a dim factor up
    #V *= np.sqrt(dim, dtype = V.dtype) 
    return V

def _check_real(V, eps = 1e-8):
    if np.max(np.abs(np.imag(V))) > eps:
        return False
    else:
        return True 

def _center_kernel(kernel, X, Y, D, averaged_indices):
    K = kernel(X, Y) 
    K_Y = kernel(Y, D).sum(1).squeeze() #Vector
    K_X = kernel(X, D).sum(1).squeeze() #Vector
    K_D = kernel(D, D).sum(0).squeeze().sum(0) #Scalar
    scale = np.array(K_X.shape[0]**-1).astype(K_X.dtype)
    K = aslinearoperator(K)
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

    return LinearOperator(K.shape, matvec =  _matvec, dtype= K.dtype)   

def modified_QR(A, M=None):
    dim = A.shape[0]
    vecs = A.shape[1]
    if M is None:
        M = identity(dim, dtype= A.dtype)
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