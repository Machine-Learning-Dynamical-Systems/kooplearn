import numpy as np
from sympy import Lambda
try:
    from pykeops.numpy import Vi, Vj, Pm
except ImportError:
    __has_keops__ = False
    Vi = Vj = Pm = lambda x: x
else:
    __has_keops__ = True  

keops_import_error = ImportError("KeOps is required for this functionality.")

def to_lazy(X, Y):
    """Utility function to convert X and Y to lazy arrays.

    Args:
        X (ndarray): Array of dimension NxD.
        Y (ndarray): Array of dimension NxD.

    Returns:
        (x, y): Tuple of lazy keops arrays.
    """ 
    x = Vi(np.ascontiguousarray(X))
    if Y is not None:
        y = Vj(np.ascontiguousarray(Y))
    else:
        y = Vj(np.ascontiguousarray(X))
    return x, y

def lazy_cdist(X, Y = None):
    """Pairwise distance between the arrays X and Y, both of dimension NxD. Returns a pykeops lazytensor."""
    x, y = to_lazy(X , Y)
    return (((x - y) ** 2).sum(2))**(0.5)        

def lazy_cprod(X,Y = None):
    """Pairwise scalar product between the arrays X and Y, both of dimension NxD. Returns a pykeops lazytensor."""
    x, y = to_lazy(X , Y)
    return (x|y)