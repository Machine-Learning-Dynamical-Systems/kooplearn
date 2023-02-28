from abc import ABCMeta, abstractmethod
import sklearn.gaussian_process.kernels as sk_kernels
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
from scipy.sparse.linalg import aslinearoperator
from ._keops_utils import Pm, lazy_cdist, lazy_cprod, __has_keops__, keops_import_error, __has_torch__
import numpy as np

 
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
        """
        Evaluate the kernel. 
        The method called with backend == ``numpy'' returns a numpy array of shape (X.shape[0], Y.shape[0]).
        The method called with backend == ``keops'' returns an instance of a scipy linearoperator of shape (X.shape[0], Y.shape[0])
        """
    def __repr__(self):
        _r = "[" + self.__class__.__name__ + "] " 
        for k, v in self.__dict__.items():
            if k[0] == "_":
                pass
            else:
                _r += f"{k}: {v} "
        return _r

class ScalarProduct(Kernel):
    def __call__(self, X, Y = None, backend = 'numpy'):
        """Return the Kernel matrix k(x,y) := <Phi(x),Phi(y)>. Different samples are batched on the first dimension of X and Y."""
        phi_X = self.__feature_map__(X)

        if Y is None:
            phi_Y = phi_X
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
    def __feature_map__(self, X):
        """Evaluate the feature map. The output should be a numpy array in _any_ case.""" 
        return X #Linear Kernel 

if __has_torch__:
    import torch
    from torch.nn import Module
    class TorchScalarProduct(Module):
        def __init__(self, feature_map):
            super().__init__()
            self.__feature_map__ = feature_map     
        def forward(self, X, Y=None, backend='numpy'):
            """
                Calculations are performed by pytorch, but if backend == 'numpy', numpy arrays are returned for consistency. 
                If backend == 'keops', an instance of a scipy linearoperator is returned instead.
            """
            with torch.no_grad():
                phi_X = self.__feature_map__(X)
                if Y is None:
                    phi_Y = phi_X
                else:
                    phi_Y = self.__feature_map__(Y)
                if backend == 'numpy':
                    return self.__to_numpy__(phi_X@(phi_Y.T))
                else:
                    return lazy_cprod(phi_X, phi_Y, backend='torch') #Backend == 'keops'
        def cov(self, X, Y = None):
            with torch.no_grad():
                phi_X = self.__feature_map__(X)
                if Y is None:
                    return self.__to_numpy__(phi_X.T@phi_X)
                else:
                    if X.shape[0] != Y.shape[0]:
                        raise ValueError("Shape mismatch: cross-covariances can be computed only if X.shape[0] == Y.shape[0] ")
                    phi_Y = self.__feature_map__(Y)
                    return self.__to_numpy__(phi_X.T@phi_Y)
        def __to_numpy__(self, tensor):
            return tensor.cpu().detach().numpy()

class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale 
        self._scikit_kernel =  sk_kernels.RBF(self.length_scale)  
        
    def __call__(self, X, Y=None, backend='numpy'):
        backend = parse_backend(backend)
        if backend == 'keops':   
            K = (-(lazy_cdist(X,Y)** 2) / (2*(Pm(self.length_scale)**2))).exp()
            return aslinearoperator(K)
        else:
            return self._scikit_kernel(X, Y)

class ExpSineSquared(Kernel):
    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self._scikit_kernel = sk_kernels.ExpSineSquared(length_scale=self.length_scale, periodicity=self.periodicity, length_scale_bounds='fixed', periodicity_bounds='fixed')
    def __call__(self, X, Y=None, backend='numpy'):
        backend = parse_backend(backend)
        if backend == 'keops':
            periodic = ((np.pi*lazy_cdist(X,Y)/Pm(self.periodicity)).sin()).square()
            K = (-(2*periodic)/((Pm(self.length_scale)**2))).exp()
            return aslinearoperator(K)
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
                K = (-D).exp()
            elif abs(self.nu - 1.5) <= 1e-12: #Once differentiable functions
                D *= sqrt(3)
                K = (1 + D)*((-D).exp())
            elif abs(self.nu - 2.5) <= 1e-12: #Twice differentiable functions
                D *= sqrt(5)
                K = (1 + D + (D**2)/3)*((-D).exp())
            else:
                raise(ValueError(f"Supported nu parameters are 0.5, 1.5, 2.5, while self.nu={self.nu}"))
            return aslinearoperator(K)
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
                return aslinearoperator(inner)
            elif self.degree == 2:
                return aslinearoperator(inner.square())
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