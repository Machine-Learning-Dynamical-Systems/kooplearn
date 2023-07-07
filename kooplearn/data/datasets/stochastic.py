from typing import Optional
from kooplearn.data.datasets.misc import DataGenerator, DiscreteTimeDynamics, LinalgDecomposition
import numpy as np
from numpy.typing import ArrayLike
import scipy
from scipy.integrate import quad
from scipy.stats import beta
from scipy.special import binom
from scipy.stats.sampling import NumericalInversePolynomial
from tqdm import tqdm
import math
import sdeint  # Optional dependency


class MockData(DataGenerator):
    def __init__(self, num_features: int = 50, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)
        self.num_features = num_features

    def generate(self, X0: ArrayLike, T: int = 1):
        return self.rng.random((T + 1, self.num_features))


class LinearModel(DiscreteTimeDynamics):
    def __init__(self, A: ArrayLike, noise: float = 0., rng_seed: Optional[int] = None):
        self.A = A
        self.noise = noise
        self.rng = np.random.default_rng(rng_seed)

    def _step(self, X: ArrayLike):
        return self.A @ X + self.noise * self.rng.standard_normal(size=X.shape)


# Noisy Logistic Map
class CosineDistribution:
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)

    def pdf(self, x):
        return self.C_N * ((np.cos(np.pi * x)) ** self.N)


class LogisticMap(DiscreteTimeDynamics):
    def __init__(self, r: float = 4.0, N: Optional[int] = None, rng_seed: Optional[int] = None):
        self._noisy = False
        self.ndim = 1
        self.rng_seed = rng_seed
        self.r = r
        if N is not None:
            assert N % 2 == 0
            # Noisy case
            self._noisy = True
            self.N = N
            self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)
            self._evals, self._PF_largest_evec, self._Koop_evecs = self._transfer_matrix_eig_process()
            self._urng = np.random.default_rng(rng_seed)
            self._rng = NumericalInversePolynomial(self, domain=(0, 1), random_state=self._urng)
            self._noise_dist = CosineDistribution(N)
            self._noise_rng = NumericalInversePolynomial(
                self._noise_dist, domain=(-0.5, 0.5), mode=0, random_state=self._urng)
        else:
            # Noiseless case
            pass

    def eig(self, num_right_fn_evals: Optional[int] = None):
        assert self._noisy, "Eigenvalue decomposition is only available for the noisy Logistic Map"
        if num_right_fn_evals is not None:
            _x = np.linspace(0, 1, num_right_fn_evals)
            _x = _x[:, None]

            phi_X = np.zeros((num_right_fn_evals, self.N + 1))
            for n in range(self.N + 1):
                phi_X[:, n] = self._feature(self.map(_x), n)[:, 0]

            ref_evd = LinalgDecomposition(
                self._evals,
                _x,
                phi_X @ self._Koop_evecs
            )
            return ref_evd
        else:
            return self._evals

    def _step(self, X_0: ArrayLike):
        return self.map(X_0, noisy=self._noisy)

    def pdf(self, x):
        if self._noisy:
            if np.isscalar(x):
                y = 0
            else:
                y = np.zeros(x.shape)
            for i in range(self.N + 1):
                y += self._feature(x, i) * self._PF_largest_evec[i]
            return np.abs(y)
        else:
            return beta(0.5, 0.5).pdf(x)

    def rvs(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._rng.rvs(size)
        else:
            return beta(0.5, 0.5).rvs(size=size)

    def noise(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._noise_rng.rvs(size)
        else:
            raise ValueError("This method not needed for noiseless case")

    def _transfer_matrix(self):
        if self._noisy:
            N = self.N
            eps = 1e-10
            A = np.zeros((N + 1, N + 1))
            for i in tqdm(range(N + 1), desc='Init: Transfer matrix'):
                for j in range(N + 1):
                    alpha = lambda x: self._feature(self.map(x), i)
                    beta = lambda x: self._feature(x, j)
                    f = lambda x: alpha(x) * beta(x)
                    q = quad(f, 0, 1, epsabs=eps, epsrel=eps)
                    A[i, j] = q[0]
            return A
        else:
            raise ValueError("This method not needed for noiseless case")

    def _transfer_matrix_eig_process(self):
        if self._noisy:
            A = self._transfer_matrix()
            self._A = A
            ev, lv, rv = scipy.linalg.eig(A, left=True, right=True)
            invariant_eig_idx = None
            for idx, v in enumerate(ev):
                if np.isreal(v):
                    if np.abs(v - 1) < 1e-10:
                        invariant_eig_idx = idx
                        break
            if invariant_eig_idx is None:
                raise ValueError("No invariant eigenvalue found")
            PF_largest_evec = rv[:, invariant_eig_idx]
            if not np.all(np.isreal(PF_largest_evec)):
                print(
                    f"Largest eigenvector is not real, largest absolute imaginary part is "
                    f"{np.abs(np.imag(PF_largest_evec)).max()}. Forcing it to be real.")
            return ev, np.real(PF_largest_evec), lv

        else:
            raise ValueError("This method not needed for noiseless case")

    def _feature(self, x, i):
        if self._noisy:
            N = self.N
            C_N = self.C_N
            return ((np.sin(np.pi * x)) ** (N - i)) * ((np.cos(np.pi * x)) ** i) * np.sqrt(binom(N, i) * C_N)
        else:
            raise ValueError("This method not needed for noiseless case")

    def map(self, x, noisy=False):
        if noisy:
            y = self.r * x * (1 - x)
            if np.isscalar(x):
                xi = self.noise(1)[0]
            else:
                xi = self.noise(x.shape)
            return np.mod(y + xi, 1)
        else:
            return self.r * x * (1 - x)


class MullerBrownPotential(DataGenerator):
    def __init__(self, kt: float = 1.5e3, rng_seed: Optional[int] = None):
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.A = np.array([-200, -100, -170, 15])
        self.X = np.array([1, 0, -0.5, -1])
        self.Y = np.array([0, 0.5, 1.5, 1])
        self.kt = kt
        self.rng = np.random.default_rng(rng_seed)

    def generate(self, X0: ArrayLike, T: int = 1):
        tspan = np.arange(0, 0.1 * T, 0.1)
        result = sdeint.itoint(self.neg_grad_potential, self.noise_term, X0, tspan, self.rng)
        return result

    def potential(self, X: ArrayLike):
        t1 = X[0] - self.X
        t2 = X[1] - self.Y
        tinexp = self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2)
        return np.dot(self.A, np.exp(tinexp))

    def neg_grad_potential(self, x: ArrayLike, t: ArrayLike):
        """
        A_j * exp[a_j * (x1^2 - 2 * x1 * X_j) + b_j * (x1 * x2 - x1 * Y_j)]

        -> grad inner exp:
            a_j * (2 * x1 - 2 * X_j) + b_j * (x2 - Y_j)
        """
        t1 = x[0] - self.X
        t2 = x[1] - self.Y
        grad_inner_exp_x1 = 2 * self.a * t1 + self.b * t2
        grad_inner_exp_x2 = 2 * self.c * t2 + self.b * t1
        tinexp = np.exp(self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2))

        return -np.array([
            np.dot(self.A, tinexp * grad_inner_exp_x1),
            np.dot(self.A, tinexp * grad_inner_exp_x2),
        ]) / self.kt

    def noise_term(self, x: ArrayLike, t: ArrayLike):
        return np.diag([math.sqrt(2 * 1e-2), math.sqrt(2 * 1e-2)])


class LangevinTripleWell1D(DiscreteTimeDynamics):
    def __init__(self, gamma: float = 0.1, kt: float = 1.0, dt: float = 1e-4, rng_seed: Optional[int] = None):
        self.gamma = gamma
        self._inv_gamma = (self.gamma) ** -1
        self.kt = kt
        self.rng = np.random.default_rng(rng_seed)
        self.dt = dt

    def _step(self, X: ArrayLike):
        F = self.force_fn(X)
        xi = self.rng.standard_normal(X.shape)
        dX = F * self._inv_gamma + np.sqrt(2.0 * self.kt * self.dt * self._inv_gamma) * xi
        return X + dX

    def force_fn(self, x: ArrayLike):
        return -1. * (-128 * np.exp(-80 * ((-0.5 + x) ** 2)) * (-0.5 + x) - 512 * np.exp(-80 * (x ** 2)) * x + 32 * (
                    x ** 7) - 160 * np.exp(-40 * ((0.5 + x) ** 2)) * (0.5 + x))
