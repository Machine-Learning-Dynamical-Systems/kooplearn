import numpy as np
from scipy.linalg import cho_factor, cho_solve

class NoisyLinear():
    def __init__(self, stability=0.5, A = None, ndim = 10, seed = None):
        
        self.rng = np.random.default_rng(seed)
        if A is not None:
            assert len(A.shape) == 2, f"The variable A should be a matrix, while it was provided an array of shape {A.shape}"
            assert A.shape[0] == A.shape[1], f"The provided matrix A is not a square matrix"
            self.A = A
            self.ndim = A.shape[0]
        else:
            self.A = self.rng.random((ndim, ndim))
            self.ndim = ndim
            
        self.stability_parameter = stability # \in [0, 1), 0 being maximally stable and 1 being unstable.
        #Rescale A to the desired stability condition
        nrm_A = np.linalg.norm(self.A, ord=2)
        self.A *=np.sqrt(self.stability_parameter)/nrm_A
        self.inverse_invariant_covariance = np.eye(self.ndim) - (self.A)@(self.A.T)
        self.invariant_covariance = np.linalg.inv(self.inverse_invariant_covariance) #Ugly
            
    def sample(self, size=1, scale_output = True, iid=True):

        assert np.isscalar(size), "The size variable should be a scalar integer"
        if iid:
            X = self.rng.multivariate_normal(np.zeros(self.ndim), self.invariant_covariance, size = size)
            Y = X@(self.A.T) + self.rng.multivariate_normal(np.zeros(self.ndim), np.eye(self.ndim), size=size)     
        else:
            _raw = np.zeros((size + 1, self.ndim))
            _raw[0] = self.rng.multivariate_normal(np.zeros(self.ndim), self.invariant_covariance)
            for i in range(1, size + 1):
                _raw[i] = self.A@_raw[i - 1] + self.rng.multivariate_normal(np.zeros(self.ndim), np.eye(self.ndim))
            X = _raw[:-1]
            Y = _raw[1:]

        if scale_output:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        return X, Y
