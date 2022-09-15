from abc import ABCMeta, abstractmethod
import sklearn.gaussian_process.kernels as sk_kernels
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
from ._keops_utils import Pm, lazy_cdist, lazy_cprod, __has_keops__, keops_import_error
from math import sqrt

def parse_backend(backend):
    if backend == 'keops':
        if __has_keops__:
            return backend
        else:
            raise keops_import_error
    elif backend == 'numpy':
        return backend
    else:
        raise ValueError('Invalid backend. Allowed values are \'numpy\' and \'keops\'.')

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, X, Y=None, backend='numpy'):
        """Evaluate the kernel."""

class FiniteDimensionalKernel(Kernel):
    def __call__(self, X, Y = None, backend = 'numpy'):
        """Return the Kernel matrix k(x,y) := <Phi(x),Phi(y)>. Different samples are batched on the first dimension of X and Y."""
        phi_X = self.__feature_map__(X)
        if Y is None:
            phi_Y = None
            if backend == 'numpy':
                return phi_X@(phi_X.T)
        else:
            phi_Y = self.__feature_map__(Y)
            if backend == 'numpy':
                return phi_X@(phi_Y.T)
        return lazy_cprod(phi_X, phi_Y) #Backend == 'keops'

    def cov(self, X, Y = None):
        phi_X = self.__feature_map__(X)
        if Y is None:
            return phi_X.T@phi_X
        else:
            if X.shape[0] != Y.shape[0]:
                raise ValueError("Shape mismatch: cross-covariances can be computed only if X.shape[0] == Y.shape[0] ")
            phi_Y = self.__feature_map__(Y)
            return phi_X.T@phi_Y
    @abstractmethod
    def __feature_map__(self, X, backend='numpy'):
        """Evaluate the feature map. The output should be a numpy array in _any_ case.""" 
        return X #Linear Kernel 

class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale 
        self._scikit_kernel =  sk_kernels.RBF(self.length_scale)  
        
    def __call__(self, X, Y=None, backend='numpy'):
        backend = parse_backend(backend)
        if backend == 'keops':   
            return (-(lazy_cdist(X,Y)** 2) / (2*(Pm(self.length_scale)**2))).exp()
        else:
            return self._scikit_kernel(X, Y)
class Matern(Kernel):
    def __init__(self, nu=1.5, length_scale=1.0):
        self.nu = nu
        self.length_scale = length_scale
        self._scikit_kernel =  sk_kernels.Matern(self.length_scale, nu=self.nu)  

    def __call__(self, X, Y=None, backend='numpy'):
        backend = parse_backend(backend)  
        if backend == 'keops':  
            D = lazy_cdist(X,Y)/Pm(self.length_scale)
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

    def __call__(self, X, Y=None, backend='numpy'):
        backend = parse_backend(backend)  
        if self.gamma is None:
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = self.gamma
        
        if backend == 'keops':
            inner = Pm(self.coef0) + lazy_cprod(X,Y)*Pm(_gamma)
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