import numpy as np
from numpy.typing import ArrayLike
import scipy.integrate 
from kooplearn.data.datasets.misc import DataGenerator

class Duffing(DataGenerator):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def generate(self, X0: ArrayLike, T:int = 1):
        raise NotImplementedErrorot


class Lorenz63(DataGenerator):
    def __init__(self, sigma=10, mu=28, beta=8/3, dt=0.01):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta  
        self.dt = dt
        self.M_lin = np.array([[-self.sigma, self.sigma, 0], [self.mu, 0, 0], [0, 0, -self.beta]])
    
    def generate(self, X0: ArrayLike, T:int = 1):
        sim_time = self.dt*(T + 1)
        t_eval = np.linspace(0, sim_time, T + 1, endpoint=True)
        t_span = (0, t_eval[-1])
        sol = scipy.integrate.solve_ivp(self.D, t_span, X0, t_eval=t_eval, method='RK45')
        return sol.y.T

    def D(self, t, x):
        dx = self.M_lin @ x
        dx[1] -= x[2]*x[0]
        dx[2] += x[0]*x[1]
        return dx
