from turtle import back
from scipy.sparse.linalg import eigs, aslinearoperator, LinearOperator, cg
from scipy.linalg import eig
from scipy.sparse import identity
import numpy as np
from warnings import warn

def reduced_rank_regression(data, evolved_data, kernel, rank, regularizer=1e-8, center_kernel = False):
    if center_kernel:
        averaged_indices_evolved = (True, True) 
        averaged_indices_cross = (True, False)
    else:
        averaged_indices_evolved = (False, False)
        averaged_indices_cross = (False, False)
    
    data_kernel = _center_kernel(kernel, data, data, data, averaged_indices=(False, False))
    evolved_data_kernel = _center_kernel(kernel, evolved_data, evolved_data, data, averaged_indices=averaged_indices_evolved)
    cross_kernel = _center_kernel(kernel, evolved_data, data, data, averaged_indices=averaged_indices_cross)
   
    V_r = _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer)
    V_r = aslinearoperator(V_r)
    dim = data.shape[0]
    scale = np.array(dim**-1).astype(data_kernel.dtype)
    
    
    U = (V_r.T)@cross_kernel
    U.dtype = V_r.dtype
    V = scale*(evolved_data_kernel@V_r)
    V.dtype = V_r.dtype
    B = data_kernel + aslinearoperator(identity(dim, dtype=U.dtype)*(regularizer*dim))
    return eigs(V@U, rank, B)

def reduced_rank_regression_new(data, evolved_data, kernel, rank, regularizer=1e-8, center_kernel = False):
    if center_kernel:
        averaged_indices_evolved = (True, True) 
        averaged_indices_cross = (False, True)
    else:
        averaged_indices_evolved = (False, False)
        averaged_indices_cross = (False, False)
    
    data_kernel = _center_kernel(kernel, data, data, data, averaged_indices=(False, False))
    evolved_data_kernel = _center_kernel(kernel, evolved_data, evolved_data, data, averaged_indices=averaged_indices_evolved)
    cross_kernel = _center_kernel(kernel, data, evolved_data, data, averaged_indices=averaged_indices_cross)
   
    V_r = _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer)
    V_r_imag_norm = np.max(np.abs(np.imag(V_r)))
    
    if V_r_imag_norm  > 1e-8:
        print(f"Imag ~ {np.imag(V_r).mean()} ± {np.imag(V_r).std()} ")
        warn(f"V_r projector is not real. Imaginary part is as high as {V_r_imag_norm}")
    
    V_r = np.real(V_r)
    dim = data.shape[0]
    scale = np.array(dim**-1).astype(data_kernel.dtype)

    V = scale*evolved_data_kernel.matmat(V_r)
    V.dtype = V_r.dtype

    U = cross_kernel.matmat(V_r)
    U = U.T
    U.dtype = V_r.dtype

    K_reg = data_kernel + aslinearoperator(identity(dim, dtype=U.dtype)*(regularizer*dim))
    
    C = np.empty_like(V)
    for r in range(V.shape[1]):
        w = V[:, r]
        C[:,r], info = cg(K_reg, w, tol=1e-8)
        if info > 0:
            warn("Convergence to tolerance not achieved")
        elif info < 0:
            warn("Illegal input or breakdown")
        else:
            pass

    C = U@C
    vals, vecs =  eig(C)
    return vals, vecs, U

def _get_low_rank_projector(data_kernel, evolved_data_kernel, rank, regularizer):
    dim = data_kernel.shape[0]
    scale = np.array(dim**-1).astype(data_kernel.dtype)
    _rescaled_evolved_data_kernel = scale*evolved_data_kernel
    _rescaled_evolved_data_kernel.dtype = evolved_data_kernel.dtype

    A = data_kernel@_rescaled_evolved_data_kernel
    B = data_kernel + aslinearoperator(identity(dim, dtype=A.dtype)*(regularizer*dim))
    _, V = eigs(A, rank, B)
    return modified_Gram_Schmidt(V, _rescaled_evolved_data_kernel)

def _center_kernel(kernel, X, Y, D, averaged_indices):
    K = kernel(X, Y) 
    K_Y = kernel(Y, D).sum(1).squeeze() #Vector
    K_X = kernel(X, D).sum(1).squeeze() #Vector
    K_D = kernel(D, D).sum(0).squeeze().sum(0) #Scalar
    
    K = aslinearoperator(K)
    if averaged_indices != (False, False):
        matvec = lambda w: _matvec(w, K_X, K_Y, K_D, averaged_indices)
        centering = LinearOperator(K.shape, matvec =  matvec)
        K = K + centering
    return K

def _matvec(w, K_X, K_Y, K_D, averaged_indices):
    w = w.squeeze()
    W = w.sum()
    scale = np.array(w.shape[0]**-1).astype(K_X.dtype)

    if averaged_indices == (True, False):
        return np.full_like(w, -np.dot(w, K_Y)*scale)
    elif averaged_indices == (False, True):
        return -K_X*W*scale
    else:
        #Default choice average both indices
        return (-K_X*W - np.dot(w, K_Y) + W*K_D*scale)*scale

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