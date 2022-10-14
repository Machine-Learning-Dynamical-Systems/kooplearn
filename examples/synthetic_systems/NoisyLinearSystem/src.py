import numpy as np
import scipy.integrate

class Lorenz63():
    def __init__(self, sigma=1, A = None, ndim = 10, seed = None):
        
        self.rng = np.random.default_rng(seed)
        if A == None:
            self.A = self.rng.random((ndim, ndim))
            self.ndim = ndim
        else:
            assert len(A.shape) == 2, f"The variable A should be a matrix, while it was provided an array of shape {A.shape}"
            assert A.shape[0] == A.shape[1], f"The provided matrix A is not a square matrix"
            self.A = A
            self.nim = A.shape[0]

        self.sigma = sigma
            
    def sample(self, x0 = None, size=1, scale_output = True):
        pass

        x = x.T
        y = y.T
        
        if scale_output:
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            y = (y - y.mean(axis=0)) / y.std(axis=0)

        return x, y