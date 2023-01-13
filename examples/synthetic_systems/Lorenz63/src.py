import numpy as np
import scipy.integrate

class Lorenz63():
    def __init__(self, sigma=10, mu=28, beta=8/3, dt=0.01, seed = None):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta  
        self.dt = dt
        self.M_lin = np.array([[-self.sigma, self.sigma, 0], [self.mu, 0, 0], [0, 0, -self.beta]])
        
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.random(3)
        
        sol = scipy.integrate.solve_ivp(self.D, (0,100), self.state, method='RK45')
        self.state = sol.y[:,-1]

        self.ndim = 3
    
    def sample(self, size=1, scale_output = True):
        if np.isscalar(size):
            size = (size + 1, 1) #Adding + 1 to the size to split x and y later
        
        sim_time = self.dt*size[0]
        t_eval = np.linspace(0, sim_time, size[0], endpoint=False)
        t_span = (0, t_eval[-1])

        sol = scipy.integrate.solve_ivp(self.D, t_span, self.state, t_eval=t_eval, method='RK45')
        x = sol.y[:,:-1]
        y = sol.y[:,1:]
        
        self.state = y[:,-1] #Update current state for next sample

        x = x.T
        y = y.T
        
        if scale_output:
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            y = (y - y.mean(axis=0)) / y.std(axis=0)

        return x, y

    def D(self, t, x):
        dx = self.M_lin @ x
        dx[1] -= x[2]*x[0]
        dx[2] += x[0]*x[1]
        return dx
