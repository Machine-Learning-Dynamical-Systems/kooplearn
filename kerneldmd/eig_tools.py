import numpy as np

def companion_matrix(alpha):
    assert alpha.ndim == 1, "alpha should be a vector"
    n = alpha.shape[0]
    S = np.zeros((n,n), dtype = alpha.dtype)
    S[1:,:-1] = np.eye(n-1)
    S[:,-1] = alpha
    return S