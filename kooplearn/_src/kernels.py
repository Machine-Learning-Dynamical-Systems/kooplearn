import abc
import sklearn.gaussian_process.kernels as sk_kernels
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
import numpy as np
import torch
from torch.nn import Module

class BaseKernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, Y=None):
        """
        Evaluate the kernel. 
        The method called with backend == ``numpy'' returns a numpy array of shape (X.shape[0], Y.shape[0]).
        """
    
    @property
    @abc.abstractmethod
    def is_inf_dimensional(self):
        pass

    def __repr__(self):
        _r = "[" + self.__class__.__name__ + "] " 
        for k, v in self.__dict__.items():
            if k[0] == "_":
                pass
            else:
                _r += f"{k}: {v} "
        return _r

class ScalarProduct(BaseKernel):
    #A base class for scalar product kernels, to be subclassed specifying the feature map. !! Covariance matrices are averaged, but kernels are not!!
    def __call__(self, X, Y = None):
        """Return the Kernel matrix k(x,y) := <Phi(x),Phi(y)>. Different samples are batched on the first dimension of X and Y."""
        phi_X = self.__feature_map__(X)
        if Y is None:
            phi_Y = phi_X
        else:
            phi_Y = self.__feature_map__(Y)
        return phi_X@(phi_Y.T)
    
    @property
    def is_inf_dimensional(self):
        return False
    
    def cov(self, X, Y = None):
        phi_X = self.__feature_map__(X)
        if Y is None:
            c = phi_X.T@phi_X
            c *= np.true_divide(1, X.shape[0])
            return c
        else:
            if X.shape[0] != Y.shape[0]:
                raise ValueError("Shape mismatch: cross-covariances can be computed only if X.shape[0] == Y.shape[0] ")
            phi_Y = self.__feature_map__(Y)
            c = phi_X.T@phi_Y
            c *= np.true_divide(1, X.shape[0])
            return c
        
    @abc.abstractmethod
    def __feature_map__(self, X):
        """Evaluate the feature map. The output should be a numpy array in _any_ case.""" 
        return X #Linear Kernel 

class TorchScalarProduct(BaseKernel, Module):
    def __init__(self, feature_map: Module):
        super().__init__()
        self.__feature_map__ = feature_map
     
    def forward(self, X, Y=None):
        """
            Calculations are internally performed by pytorch but numpy arrays are always accepted and returned. 
        """
        X = torch.asarray(X)
        with torch.no_grad():
            phi_X = self.__feature_map__(X)
            if Y is None:
                phi_Y = phi_X
            else:
                Y = torch.asarray(Y)
                phi_Y = self.__feature_map__(Y)
        return self.__to_numpy__(phi_X@(phi_Y.T))
    
    @property
    def is_inf_dimensional(self):
        return False
            
    def cov(self, X, Y = None):
        with torch.no_grad():
            phi_X = self.__feature_map__(X)
            if Y is None:
                c =  self.__to_numpy__(phi_X.T@phi_X)
                c *= np.true_divide(1, list(X.shape)[0])
                return c
            else:
                if X.shape[0] != Y.shape[0]:
                    raise ValueError("Shape mismatch: cross-covariances can be computed only if X.shape[0] == Y.shape[0] ")
                phi_Y = self.__feature_map__(Y)
                c =  self.__to_numpy__(phi_X.T@phi_Y)
                c *= np.true_divide(1, list(X.shape)[0])
                return c
    
    def __to_numpy__(self, tensor):
        return tensor.cpu().detach().numpy()

class RBF(BaseKernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale 
        self._scikit_kernel =  sk_kernels.RBF(self.length_scale)
    
    @property
    def is_inf_dimensional(self):
        return True
        
    def __call__(self, X, Y=None):
        return self._scikit_kernel(X, Y)

class ExpSineSquared(BaseKernel):
    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self._scikit_kernel = sk_kernels.ExpSineSquared(length_scale=self.length_scale, periodicity=self.periodicity, length_scale_bounds='fixed', periodicity_bounds='fixed')
    @property
    def is_inf_dimensional(self):
        return True
    def __call__(self, X, Y=None):
        return self._scikit_kernel(X, Y)

class Matern(BaseKernel):
    def __init__(self, nu=1.5, length_scale=1.0):
        self.nu = nu
        self.length_scale = length_scale
        self._scikit_kernel =  sk_kernels.Matern(self.length_scale, nu=self.nu)  
    @property
    def is_inf_dimensional(self):
        return True
    
    def __call__(self, X, Y=None):
        return self._scikit_kernel(X, Y)
    
class Poly(BaseKernel):
    """
        Polynomial Kernel
        K(X, Y) = (gamma <X, Y> + coef0)^degree
    """
    def __init__(self, degree=3, gamma=None, coef0=1):
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
    
    @property
    def is_inf_dimensional(self):
        return False

    def __call__(self, X, Y=None):
        return sk_poly(X, Y, degree = self.degree, gamma = self.gamma, coef0 = self.coef0)  
                     
class Linear(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=1, gamma=gamma, coef0=coef0)

class Quadratic(Poly):
    def __init__(self, gamma=None, coef0=1):
        super().__init__(degree=2, gamma=gamma, coef0=coef0)