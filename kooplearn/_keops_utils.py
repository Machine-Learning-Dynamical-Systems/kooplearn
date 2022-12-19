import numpy as np
try:
    from pykeops.numpy import Vi, Vj, Pm
except ImportError:
    __has_keops__ = False
    Vi = Vj = Pm = lambda x: x
else:
    __has_keops__ = True

try:
    import torch 
except ImportError:
    __has_torch__ = False
else:
    __has_torch__ = True

if __has_keops__ and __has_torch__:
    from pykeops.torch import Vi as Vi_torch
    from pykeops.torch import Vj as Vj_torch
    

keops_import_error = ImportError("KeOps is required for this functionality.")

def to_lazy(X, Y, backend = 'numpy'):
    """Utility function to convert X and Y to lazy arrays.

    Args:
        X (ndarray): Array of dimension NxD.
        Y (ndarray): Array of dimension NxD.

    Returns:
        (x, y): Tuple of lazy keops arrays.
    """ 
    if backend == 'numpy':
        x = Vi(np.ascontiguousarray(X))
        if Y is not None:
            y = Vj(np.ascontiguousarray(Y))
        else:
            y = Vj(np.ascontiguousarray(X))
        return x, y
    elif backend == 'torch':
        assert torch.is_tensor(X)
        x = Vi_torch(X.contiguous())
        if Y is not None:
            assert torch.is_tensor(Y)
            y = Vj_torch(Y.contiguous())
        else:
            y = Vj_torch(torch.clone(X).contiguous())
        return x, y
    else:
        raise ValueError(f"Unknown backend ``{backend}''. Accepted values are ``numpy'' or ``torch''.")

def lazy_cdist(X, Y = None, backend='numpy'):
    """Pairwise distance between the arrays X and Y, both of dimension NxD. Returns a pykeops lazytensor."""
    x, y = to_lazy(X , Y, backend=backend)
    return (((x - y) ** 2).sum(2))**(0.5)        

def lazy_cprod(X,Y = None, backend='numpy'):
    """Pairwise scalar product between the arrays X and Y, both of dimension NxD. Returns a pykeops lazytensor."""
    x, y = to_lazy(X , Y, backend=backend)
    return (x|y)