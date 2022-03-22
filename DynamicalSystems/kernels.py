from abc import ABCMeta, abstractmethod
from scipy.sparse.linalg import aslinearoperator
from pykeops.numpy import Vi, Vj, Pm
from math import sqrt
import numpy as np

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, X, Y=None):
        """Evaluate the kernel."""

    def to_numpy(self, X, Y=None):
        _lazy_kernel = self.__call__(X,Y)
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
        self.length_scale = Pm(length_scale)    
        
    def __call__(self, X, Y=None):   
        return (-(self.cdist(X,Y)** 2) / (2*(self.length_scale**2))).exp()

class Matern(Kernel):
    def __init__(self, nu=1.5, length_scale=1.0):
        self.nu = nu
        self.length_scale = Pm(length_scale)

    def __call__(self, X, Y=None):

        D = self.cdist(X,Y)/self.length_scale
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

class Poly(Kernel):
    """
        Polynomial Kernel
        K(X, Y) = (gamma <X, Y> + coef0)^degree
    """
    def __init__(self, degree=3, gamma=None, coef0=1):
        self.gamma = gamma
        self.coef0 = Pm(coef0)
        self.degree = degree

    def __call__(self, X, Y=None):
        if self.gamma is None:
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = self.gamma
            
        _gamma = Pm(_gamma)

        inner = self.coef0 + self.cprod(X,Y)*_gamma
        if self.degree == 1:
            return inner
        elif self.degree == 2:
            return inner.square()
        else:
            raise NotImplementedError("Poly kernel with degree != [1,2] not implemented (because of a bug on pow function).")

class Linear(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=1, gamma=gamma, coef0=coef0)

class Quadratic(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=2, gamma=gamma, coef0=coef0)