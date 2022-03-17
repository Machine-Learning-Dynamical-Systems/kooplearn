from abc import ABCMeta, abstractmethod
from pykeops.numpy import Vi, Vj, Pm
from math import sqrt
from torch import cdist as torch_cdist

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, X, Y=None):
        """Evaluate the kernel."""


    def _to_lazy(self, X, Y):
        x = Vi(X)
        if Y is not None:
            y= Vj(Y)
        else:
            y = Vj(X)
        return x, y

    def cdist(self, X, Y= None):
        x, y = self._to_lazy(X , Y)
        return (((x - y) ** 2).sum(2))**(0.5)        

    def cprod(self, X,Y=None):
        x, y = self._to_lazy(X , Y)
        return (x*y).sum(2)

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
        self.degree = Pm(degree)

    def __call__(self, X, Y=None):
        if self.gamma is None:
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = self.gamma
            
        _gamma = Pm(_gamma)

        return (self.coef0 + self.cprod(X,Y)*_gamma)**self.degree