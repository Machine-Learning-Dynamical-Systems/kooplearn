from abc import ABCMeta, abstractmethod
from scipy.sparse.linalg import aslinearoperator
import sklearn.gaussian_process.kernels as sk_kernels
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
from pykeops.numpy import Vi, Vj, Pm
from math import sqrt
import numpy as np
from DynamicalSystems.utils import parse_backend

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, X, Y=None, backend='auto'):
        """Evaluate the kernel."""

    def to_numpy(self, X, Y=None):
        _lazy_kernel = self.__call__(X,Y, backend = 'keops')
        if Y is not None:
            if X.shape[0] != Y.shape[0]:
                raise NotImplementedError("Implemented dense representation only for squared matrices")
        Id = np.eye(_lazy_kernel.shape[0], dtype=_lazy_kernel.dtype, order ='F')
        return aslinearoperator(_lazy_kernel).matmat(Id)
    
    def _to_lazy(self, X, Y):
        x = Vi(np.ascontiguousarray(X))
        if Y is not None:
            y= Vj(np.ascontiguousarray(Y))
        else:
            y = Vj(np.ascontiguousarray(X))
        return x, y

    def cdist(self, X, Y= None):
        x, y = self._to_lazy(X , Y)
        return (((x - y) ** 2).sum(2))**(0.5)        

    def cprod(self, X,Y=None):
        x, y = self._to_lazy(X , Y)
        return (x|y)

class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale 
        self._scikit_kernel =  sk_kernels.RBF(self.length_scale)  
        
    def __call__(self, X, Y=None, backend='auto'):
        backend = parse_backend(backend, X)
        if backend == 'keops':   
            return (-(self.cdist(X,Y)** 2) / (2*(Pm(self.length_scale)**2))).exp()
        else:
            return self._scikit_kernel(X, Y)
            
class Matern(Kernel):
    def __init__(self, nu=1.5, length_scale=1.0):
        self.nu = nu
        self.length_scale = length_scale
        self._scikit_kernel =  sk_kernels.Matern(self.length_scale, nu=self.nu)  

    def __call__(self, X, Y=None, backend='auto'):
        backend = parse_backend(backend, X)  
        if backend == 'keops':  
            D = self.cdist(X,Y)/Pm(self.length_scale)
            if abs(self.nu - 0.5) <= 1e-12:
                return (-D).exp()
            elif abs(self.nu - 1.5) <= 1e-12: #Once differentiable functions
                D *= sqrt(3)
                return (1 + D)*((-D).exp())
            elif abs(self.nu - 2.5) <= 1e-12: #Twice differentiable functions
                D *= sqrt(5)
                return (1 + D + (D**2)/3)*((-D).exp())
            else:
                raise(ValueError(f"Supported nu parameters are 0.5, 1.5, 2.5, while self.nu={self.nu}"))
        else:
            return self._scikit_kernel(X, Y)
class Poly(Kernel):
    """
        Polynomial Kernel
        K(X, Y) = (gamma <X, Y> + coef0)^degree
    """
    def __init__(self, degree=3, gamma=None, coef0=1):
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        

    def __call__(self, X, Y=None, backend='auto'):
        backend = parse_backend(backend, X)  
        if self.gamma is None:
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = self.gamma
        
        if backend == 'keops':
            inner = Pm(self.coef0) + self.cprod(X,Y)*Pm(_gamma)
            if self.degree == 1:
                return inner
            elif self.degree == 2:
                return inner.square()
            else:
                raise NotImplementedError("Poly kernel with degree != [1,2] not implemented (because of a bug on pow function).")
        else:
            return sk_poly(X, Y, degree = self.degree, gamma = self.gamma, coef0 = self.coef0)
            
class Linear(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=1, gamma=gamma, coef0=coef0)

class Quadratic(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=2, gamma=gamma, coef0=coef0)