import numpy as np

class DataGenerator:
    def __init__(self):
        pass

    def generate(self, X0, T=1):
        memory = np.zeros((T+1, X0.shape))
        memory[0] = X0
        for t in T:
            memory[t+1] = self._step(memory[t])
        return memory

    def _step(self, X):
        pass

class LinearModel(DataGenerator):
    def __init__(self, phi=None, noise=0.):
        self.phi = phi
        self.noise = noise

    def _step(self, X):
        return self.phi@X + self.noise*np.random.normal(0,1,size=X.shape)

class NoisyLogisticMap(DataGenerator):
    def __init__(self, r=1., noise=0.):
        self.r = r
        self.noise = 0
    
    def _step(self, X):
        return self.r * X * (1 - X) + self.noise*np.random.normal(0,1,size=X.shape)

class Lorenz63(DataGenerator):
    def __init__(self, sigma, rho, beta):
        self.sigma=sigma
        self.rho=rho
        self.beta=beta

    def _step(self, X):
        assert X.shape == 3, 'Lorenz63 requires 3-dimensional data, {}-dimensional was given'.format(X.shape)
        x_new = X[0] + self.sigma*(X[1] - X[0])
        y_new = X[0] * (self.rho - X[2])
        z_new = X[0]* X[1] + (1-self.beta)*X[2]
        return np.array([x_new, y_new, z_new])

class Duffing(DataGenerator):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def _step(self, X):
        assert X.shape == 2, 'Duffing requires 2-dimensional data, {}-dimensional was given'.format(X.shape)
        x_new = X[1]
        y_new = -self.b * X[0] + self.a * X[1] + X[1]**3
        return np.array([x_new, y_new])