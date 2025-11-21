import numpy as np
from typing import Optional

from kooplearn.datasets.misc import DataGenerator
from kooplearn.datasets.stochastic import DiscreteTimeDynamics


class DiscreteBlackScholes(DiscreteTimeDynamics):
    """
    Discretised simulation of the Black Scholes model.

    Args:
        A: transition matrix of the drift of shape ''(d, d)''.
        sigma1: covariance matrix of the increments of the prices brownian motion, shape is ''(d,d)''.
        dt: size of the increments.

    Attributes:
       sample: samples the process over T iterations with starting point X0.
    """

    def __init__(
        self,
        A: np.ndarray,
        sigma: np.ndarray,
        dt: float,
        rng_seed: Optional[int] = None,
    ):
        self.dim = A.shape[0]
        self.A = A
        self.dt = dt
        self.sigma = sigma
        self.rng = np.random.default_rng(rng_seed)

    def _step(self, X: np.ndarray):
        return (
            X
            + self.A @ X * self.dt
            + np.sqrt(self.dt)
            * X
            * self.rng.multivariate_normal(np.zeros(self.dim), self.sigma)
        )


class DiscreteOhrnstein(DiscreteTimeDynamics):
    """
    Discretised simulation of the Ohrnstein-Ulhenbeck volatility model.

    Args:
        mu: mean of the volatility process, array of size ''d''.
        beta: driving coefficient of the volatility process, matrix of shape ''(d,d)''.
        sigma: covariance matrix of the increments of the volatility's volatility, matrix of shape ''(d,d)''.
        dt: size of the increments.

    Attributes:
       sample: samples the process over T iterations with starting point X0.
    """

    def __init__(
        self,
        mu: np.ndarray,
        beta: np.ndarray,
        sigma: np.ndarray,
        dt: float,
        rng_seed: Optional[int] = None,
    ):
        self.dim = mu.shape[0]
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(rng_seed)

    def _step(self, X: np.ndarray):
        return (
            X
            + self.beta @ (self.mu - X) * self.dt
            + np.sqrt(self.dt)
            * self.rng.multivariate_normal(np.zeros(self.dim), self.sigma)
        )


class DiscreteCIR(DiscreteOhrnstein):
    """
    Discretised simulation of the CIR volatility model.

    Args:
        mu: mean of the volatility process, array of size ''d''.
        beta: driving coefficient of the volatility process, matrix of shape ''(d,d)''.
        sigma: covariance matrix of the increments of the volatility's volatility, matrix of shape ''(d,d)''.
        dt: size of the increments.

    Attributes:
       sample: samples the process over T iterations with starting point X0.
    """

    def _step(self, X: np.ndarray):
        return (
            X
            + self.beta @ (self.mu - X) * self.dt
            + np.sqrt(self.dt)
            * np.sqrt(X)
            * self.rng.multivariate_normal(np.zeros(self.dim), self.sigma)
        )


# Heston like model (ohrnstein volatility)
class DiscreteHeston(DiscreteTimeDynamics):
    """
    Discretised simulation of the Heston model.

    Args:
        A: transition matrix of the drift of shape ''(d, d)''.
        sigma1: covariance matrix of the increments of the prices brownian motion, shape is ''(d,d)''.
        mu: mean of the volatility process, array of size ''d''.
        beta: driving coefficient of the volatility process, matrix of shape ''(d,d)''.
        sigma2: covariance matrix of the increments of the volatility's volatility, matrix of shape ''(d,d)''.
        dt: size of the increments.

    Attributes:
       sample: samples the process over T iterations with starting point X0.
    """

    def __init__(
        self,
        A: np.ndarray,
        sigma1: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        sigma2: np.ndarray,
        nu0: np.ndarray,
        dt: float,
        rng_seed: Optional[int] = None,
    ):
        self.A = A
        self.dim = A.shape[0]
        self.beta = beta
        self.mu = mu
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.nu0 = nu0
        self.dt = dt
        self.rng = np.random.default_rng(rng_seed)

    def _step(self, X: np.ndarray):
        next_X = (
            X
            + self.dt * self.A @ X
            + np.sqrt(self.dt)
            * np.sqrt(self.nu0)
            * X
            * self.rng.multivariate_normal(np.zeros(self.dim), self.sigma2)
        )
        self.nu0 += self.dt * self.beta @ (self.mu - self.nu0) + np.sqrt(
            self.dt
        ) * np.sqrt(self.nu0) * self.rng.multivariate_normal(
            np.zeros(self.dim), self.sigma1
        )
        return next_X


class Garch(DataGenerator):
    # one dimensional one lag garch model
    def __init__(self, alpha: float, beta: float, alpha0: float = 0.0):
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
