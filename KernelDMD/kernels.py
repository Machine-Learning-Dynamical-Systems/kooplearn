from abc import ABCMeta, abstractmethod
from pykeops.numpy import LazyTensor
from math import sqrt
from torch import cdist as torch_cdist

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, X, Y=None):
        """Evaluate the kernel."""

    def cdist(self, X, Y=None, backend='torch'):
        if backend == 'torch':
            if Y is None:
                return torch_cdist(X,X)
            else:
                return torch_cdist(X,Y)
        elif backend == 'keops':
            x_ = LazyTensor(X[:,None,:])
            if Y is not None:
                y_ = LazyTensor(Y[None,:,:])
            else:
                y_ = LazyTensor(X[None,:,:]) 
            return (((x_ - y_) ** 2).sum(2))**(0.5)
        else:
            raise ValueError("Supported backends are 'torch' or 'keops'")


class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X, Y=None, backend='torch'):   
        return (-(self.cdist(X,Y, backend=backend)** 2) / 2*(self.length_scale**2)).exp()

class Matern(Kernel):
    def __init__(self, nu=1.5, length_scale=1.0):
        self.nu = nu
        self.length_scale = length_scale

    def __call__(self, X, Y=None, backend='torch'):
        x_ = LazyTensor(X[:,None,:]/self.length_scale)
        if Y is not None:
            y_ = LazyTensor(Y[None,:,:]/self.length_scale)
        else:
            y_ = LazyTensor(X[None,:,:]/self.length_scale) 
        D = self.cdist(X,Y, backend=backend)
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
    """Polynomial Kernel
        K(X, Y) = (gamma <X, Y> + coef0)^degree
    """
    def __init__(self, degree=3, gamma=None, coef0=1):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if self.gamma is None:
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = self.gamma
        x_ = LazyTensor(X[:,None,:])
        if Y is not None:
            y_ = LazyTensor(Y[None,:,:])
        else:
            y_ = LazyTensor(X[None,:,:]) 
        return (self.coef0 + (x_*y_).sum(2)*_gamma)**self.degree