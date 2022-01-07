import numpy as np
from kerneldmd._kernel_helpers import pdist2, pdist_sym2, pscal, pscal_sym
__EPS__ = 1e-10
__all__ = ['Matern', 'RBF', 'Polynomial']
def Matern(*X, nu=1.5, l=1):
    #Euclidean distance matrix
    D = _dist_matrix(X, squared=False)
    if abs(nu - 0.5) <= __EPS__:
        return np.exp(-D/l)
    elif abs(nu - 1.5) <= __EPS__: #Once differentiable functions
        D *= np.sqrt(3)/l
        return (1 + D)*(np.exp(-D))
    elif abs(nu - 2.5) <= __EPS__: #Twice differentiable functions
        D *= np.sqrt(5)/l
        return (1 + D + (D**2)/3)*(np.exp(-D))
    else:
        raise(ValueError(f"nu parameter should be in [0.5, 1.5, 2.5]. Inserted nu={nu}"))

def RBF(*X, l=1):
    #Euclidean distance matrix
    D2 = _dist_matrix(X, squared=True)
    return np.exp(-D2/(2*l*l))

def Polynomial(*X, b=1, p=1, l=1):
    #Euclidean distance matrix
    D = _scal_matrix(X)/l
    return (D + b)**p

def _dist_matrix(X, squared=True):
    assert len(X) <= 2, f"Either one or two data arrays are supported, but {len(X)} were given."
    #Arrays [observations, features]
    if len(X) == 2:
        assert X[0].shape[1] == X[1].shape[1], f"Features dimension not matching in the two arrays {X[0].shape[1]} != {X[1].shape[1]}"
        dtype = X[0].dtype if X[0].dtype.itemsize >= X[1].dtype.itemsize else X[1].dtype
        if squared:
            return pdist2(X[0], X[1], dtype)
        else:
            return np.sqrt(pdist2(X[0], X[1], dtype))
    else:
        dtype = X[0].dtype
        if squared:
            return pdist_sym2(X[0], dtype)
        else:
            return np.sqrt(pdist_sym2(X[0], dtype))

def _scal_matrix(X):
    assert len(X) <= 2, f"Either one or two data arrays are supported, but {len(X)} were given."
    #Arrays [observations, features]
    if len(X) == 2:
        assert X[0].shape[1] == X[1].shape[1], f"Features dimension not matching in the two arrays {X[0].shape[1]} != {X[1].shape[1]}"
        dtype = X[0].dtype if X[0].dtype.itemsize >= X[1].dtype.itemsize else X[1].dtype
        return pscal(X[0], X[1], dtype)
    else:
        dtype = X[0].dtype
        return pscal_sym(X[0], dtype)