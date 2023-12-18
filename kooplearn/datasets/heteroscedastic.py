import numpy as np

from kooplearn.datasets.misc import DataGenerator


class Garch(DataGenerator):
    # one dimensional one lag garch model
    def __init__(self, alpha, beta, alpha0=0.0):
        self.alpha = alpha
        self.beta = beta
        self.alpha0 = alpha0

    def sample(self, X0, T=1):
        memory = np.zeros(T + 1)
        memory[0] = X0
        aux_memory = np.zeros(T + 1)

        noise = np.random.normal(0, 1, size=T)

        for t in range(T):
            aux_memory[t + 1] = (
                self.alpha0 + self.alpha * (memory[t] ** 2) + self.beta * aux_memory[t]
            )
            memory[t + 1] = noise[t] * np.sqrt(aux_memory[t + 1])

        return memory


class DMgarch(DataGenerator):
    # mutlidimensional one lag diagonal mgarch model
    def __init__(self, s, A, B):
        self.A = A
        self.B = B
        self.s = s

    def sample(self, X0, T=1):
        memory = np.zeros((T, X0.shape))
        memory[0] = X0
        H = np.zeros((X0.shape, X0.shape))

        for t in range(T):
            H = self.s + self.A * np.outer(memory[t], memory[t]) + self.B * H
            memory[t + 1] = np.linalg.cholesky(H) @ np.random.normal(
                0, 1, size=X0.shape
            )

        return memory
