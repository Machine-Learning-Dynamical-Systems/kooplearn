from scipy.sparse.linalg import eigsh, aslinearoperator
import torch
import numpy 

def _get_scaled_SVD_right_eigvectors(X, num_modes, backend):
    if backend =='torch':
        Sigma, V = torch.linalg.eigh(X)
        Sigma_r, V_r = Sigma[-num_modes:], V[:,-num_modes:]
        return V_r@torch.diag(torch.sqrt(Sigma_r)**-1)
    elif backend =='keops':
        Sigma_r, V_r = eigsh(aslinearoperator(X), k = num_modes)
        return V_r@numpy.diag(numpy.sqrt(Sigma_r)**-1)
    else:
        raise ValueError("Supported backends are 'torch' or 'keops'")

def _DMD(trajectory, evolved_trajectory, kernel, num_modes, backend):
    """DMD using truncated SVD decomposition

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    Vhat_r = _get_scaled_SVD_right_eigvectors(kernel(trajectory, backend=backend), num_modes, backend)
    Ahat = Vhat_r.T @(kernel(trajectory,evolved_trajectory, backend=backend)@Vhat_r)
    if backend =='torch':
        evals, evecs = torch.linalg.eig(Ahat)
        Vhat_r = Vhat_r.type(evecs.type()) #convert to Complex type
    elif backend =='keops':
        evals, evecs = numpy.linalg.eig(Ahat)
    else:
        raise ValueError("Supported backends are 'torch' or 'keops'")    
    return evals, Vhat_r@evecs

def DMDs(trajectory, evolved_trajectory, kernel, num_modes):
    """Sparse DMD using Lanczos iterations. Useful for large problems

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    return _DMD(trajectory, evolved_trajectory, kernel, num_modes, 'keops')

def DMD(trajectory, evolved_trajectory, kernel, num_modes):
    """Sparse DMD using Lanczos iterations. Useful for large problems

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    return _DMD(trajectory, evolved_trajectory, kernel, num_modes, 'torch')