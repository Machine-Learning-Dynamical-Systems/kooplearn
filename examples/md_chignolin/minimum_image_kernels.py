from pykeops.numpy import LazyTensor
from math import sqrt
import torch

def cdist(X, Y=None, backend='torch', unit_cell_length=1.0):
    if backend == 'torch':
        if Y is None:
            dR = torch.unsqueeze(X,0) - torch.unsqueeze(X, 1)
        else:
            dR = torch.unsqueeze(X,0) - torch.unsqueeze(Y, 1)
        return torch.sqrt(torch.sum((dR - unit_cell_length*torch.round(dR/unit_cell_length))**2, dim=2))
    elif backend == 'keops':
        x_ = LazyTensor(X[:,None,:])
        if Y is not None:
            y_ = LazyTensor(Y[None,:,:])
        else:
            y_ = LazyTensor(X[None,:,:])
        dR = x_ - y_
        return (((dR - unit_cell_length*((dR/unit_cell_length).round())) ** 2).sum(2))**(0.5)
    else:
        raise ValueError("Supported backends are 'torch' or 'keops'")

class RBF:
    def __init__(self, length_scale=1.0, unit_cell_length=1.0):
        self.length_scale = length_scale
        self.unit_cell_length = unit_cell_length
    def __call__(self, X, Y=None, backend='torch'):   
        return (-(cdist(X,Y, backend=backend, unit_cell_length=self.unit_cell_length)** 2) / (2*(self.length_scale**2))).exp()

class Matern:
    def __init__(self, nu=1.5, length_scale=1.0, unit_cell_length=1.0):
        self.nu = nu
        self.length_scale = length_scale
        self.unit_cell_length = unit_cell_length

    def __call__(self, X, Y=None, backend='torch'):
        D = self.cdist(X,Y, backend=backend, unit_cell_length=self.unit_cell_length)/self.length_scale
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
