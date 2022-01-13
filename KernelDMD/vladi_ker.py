#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:40:40 2022

@author: vkiit
"""

import numpy as np

from pykeops.numpy import LazyTensor #, generic_argmin
from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.linalg import eig


class kernel:
    def __init__(self,**kwargs):
        if not 'name' in kwargs:
            kwargs['name'] = 'rbf'
        if not 'dtype' in kwargs:
            kwargs['dtype'] = "float64"
        if not 'params' in kwargs:
            kwargs['params'] = 1
        self.dtype = kwargs['dtype']
        self.name = kwargs['name']
        self.params = kwargs['params']
        if self.name == 'rbf':
            self.matrix = self.rbf
        elif self.name == 'poly':
            self.matrix = self.poly
        elif self.name == 'matern':
            self.matrix = self.matern
        else:
            raise(ValueError("Not suported kernel type"))

    def rbf(self, x,y):
        sigma = self.params
        x_, y_ = x / sigma, y / sigma
        x_i, y_j = LazyTensor(x_[:, None, :]), LazyTensor(y_[None, :, :])
        return (-((x_i - y_j) ** 2).sum(2) / 2).exp()
    
    def poly(self, x, y):
        b, l, p = self.params
        x_i, y_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
        return (b+(x_i*y_j).sum()/l)**p
 
 
    def matern(self, x, y):
        nu, sigma =self.params
        x_, y_ = x / sigma, y / sigma
        x_i, y_j = LazyTensor(x_[:, None, :]), LazyTensor(y_[None, :, :])
        D = (((x_i - y_j) ** 2).sum(2))**(0.5)
        if abs(nu - 0.5) <= 1e-12:
            return (-D/sigma).exp()
        elif abs(nu - 1.5) <= 1e-12: #Once differentiable functions
            D *= np.sqrt(3)/sigma
            return (1 + D)*(-D).exp()
        elif abs(nu - 2.5) <= 1e-12: #Twice differentiable functions
            D *= np.sqrt(5)/sigma
            return (1 + D + (D**2)/3)*(-D).exp()
        else:
            raise(ValueError(f"nu parameter should be in [0.5, 1.5, 2.5]. Inserted nu={nu}"))
 
    def evd(self,x, k = 10):      
        return eigsh(aslinearoperator(self.matrix(x,x)), k = k)
    
    def dmd(self,X,r):
        Sigma_r, V_r = self.evd(X[:,:-1].T, k = r)
        Vhat_r = V_r @ np.diag(np.sqrt(Sigma_r))
        Ahat = Vhat_r.T @(self.matrix(X[:,:-1].T,X[:,1:].T) @ Vhat_r)

        evals , evecs = eig(Ahat) 
    
        return evals, Vhat_r@evecs