#from pykeops.torch import LazyTensor
#from math import sqrt
import numpy as np


__EPS__ = 1e-10

def matern(D, nu=1.5, l=1):
    #Assuming D = squared distance matrix
    D = np.sqrt(D)
    if abs(nu - 0.5) <= __EPS__:
        return np.exp(-D/l)
    elif abs(nu - 1.5) <= __EPS__: #Once differentiable functions
        _D = np.sqrt(3)*D/l
        return (1 + _D)*(np.exp(-_D))
    elif abs(nu - 2.5) <= __EPS__: #Twice differentiable functions
        _D = np.sqrt(5)*D/l
        return (1 + _D + (_D**2)/3)*(np.exp(-_D))
    else:
        raise(ValueError(f"nu parameter should be in [0.5, 1.5, 2.5]. Inserted nu={nu}"))

def gaussian(D , l=1):
    #Assuming D = squared distance matrix
    l = l**2
    l *=2
    return np.exp(-D/l)



def matern_keops(X1, X2, nu=1.5, l=1):
    #Assuming X = [observations, features]
    _d = LazyTensor(X1[:, None, :]) - LazyTensor(X2[None, :, :])
    D = ((_d) ** 2).sum(-1).sqrt() #Euclidean distance
    if abs(nu - 0.5) <= __EPS__:
        return (-D/l).exp()
    elif abs(nu - 1.5) <= __EPS__: #Once differentiable functions
        _D = sqrt(3)*D/l
        return (1 + _D)*((-_D).exp())
    elif abs(nu - 2.5) <= __EPS__: #Twice differentiable functions
        _D = sqrt(5)*D/l
        return (1 + D + (D**2)/3)*((-_D).exp())
    else:
        raise(ValueError(f"nu parameter should be in [0.5, 1.5, 2.5]. Inserted nu={nu}"))

def gaussian_keops(X1, X2 , l=1):
    #Assuming X = [observations, features]
    l = l**2
    l *=2
    #Assuming X = [observations, features]
    _d = LazyTensor(X1[:, None, :]) - LazyTensor(X2[None, :, :])
    D = ((_d) ** 2).sum(-1) #Euclidean squared distance
    return (-D/l).exp()