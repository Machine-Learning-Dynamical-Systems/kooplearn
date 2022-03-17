from turtle import back
from attr import evolve
from scipy.sparse.linalg import eigs, aslinearoperator, LinearOperator
from scipy.linalg import eig
from scipy.sparse import diags, identity
from pykeops.numpy import Vi
import numpy as np
from warnings import warn
    
def reduced_rank_regression(data, evolved_data, kernel, rank, regularizer=1e-8, center_kernel = False, _cumbersome_diagonalization = True):

    # Defining kernels
    data_kernel = kernel(data, data) 
    evolved_data_kernel = aslinearoperator(kernel(evolved_data, evolved_data))
    #Working with the transpose of the matrix in the notes for easier implementation
    cross_kernel = aslinearoperator(kernel(data, evolved_data))
    
    dim = data_kernel.shape[0]
    scale = np.array(dim**-1).astype(data_kernel.dtype)    

    #Correcting Kernel if needed
    if center_kernel:
        evolved_data_kernel += _center_kernel(kernel, evolved_data, evolved_data, data, averaged_indices=(True, True))
        cross_kernel += _center_kernel(kernel, data, evolved_data, data, averaged_indices=(False, True))
   
    proj = _get_low_rank_projector(aslinearoperator(data_kernel*scale.item()), evolved_data_kernel, rank, regularizer)
    
    #Neither U nor V are normalised by 1/dim because from _get_low_rank_projector we spare a factor dim
    U = cross_kernel.matmat(proj)
    U = U.T #[rank, dim]
    V = evolved_data_kernel.matmat(proj) # [dim, rank]

    if _cumbersome_diagonalization:    
        # V@(U.T)w =  λ K_reg w
        B = aslinearoperator(data_kernel) + aslinearoperator(diags(np.ones(dim, dtype=U.dtype)*(regularizer*dim)))
        return eigs(aslinearoperator(V)@aslinearoperator(U), rank, B)
    else:
        # U@K_reg^-1@V w = λw
        # C := K_reg^-1@V
        C = data_kernel.solve(Vi(V), alpha=regularizer*dim, eps=1e-12)
        vals, vecs =  eig(U@C)
        return vals, C@vecs

def _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer):
    #For the moment data_kernel = LinearOperator, evolved_data_kernel = LinearOperator
    dim = data_kernel.shape[0]

    KK_P = data_kernel@evolved_data_kernel
    tikhonov_reg = aslinearoperator(diags(np.ones(dim, dtype=KK_P.dtype)*(regularizer*dim)))
    K_reg = data_kernel + tikhonov_reg
    _, V = eigs(KK_P, rank, K_reg)
    
    #V now are orthonormal w.r.t. K_P and not K_P/dim. They should be normalised by multiplying by sqrt(dim). Let's skip this and lose a dim factor up
    #V *= np.sqrt(dim, dtype = V.dtype) 

    #Check that the eigenvectors are real (or have a global phase at most)
    if not _check_real(V):
        V_global_phase_norm = np.angle(V).std()
        if V_global_phase_norm  > 1e-8:
            print(f"Global Phase is ~ {np.around(np.angle(V).mean(), decimals = 3)} ± {np.around(V_global_phase_norm, decimals = 3)} ")
            warn(f"Computed projector is not real, chopping imaginary part.")
            V = np.real(V)
        else:
            #It has a complex phase, take absolute.
            V = np.abs(V)
    else:
        V = np.real(V)
    return modified_QR(V, evolved_data_kernel)

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
    return LinearOperator(K.shape, matvec =  _matvec)   

def modified_QR(A, M=None):
    dim = A.shape[0]
    vecs = A.shape[1]
    if M is None:
        M = identity(dim, dtype= A.dtype)
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