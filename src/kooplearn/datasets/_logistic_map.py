import numpy as np
import scipy.integrate
import scipy.special
from numpy.typing import ArrayLike
from scipy.stats.sampling import NumericalInversePolynomial
from tqdm import tqdm


def logistic_map(x: ArrayLike):
    return 4 * x * (1 - x)


def noise_features(x: ArrayLike, i: int, M: int = 10):
    cst = np.sqrt(
        np.pi * scipy.special.binom(2 * M, i) / scipy.special.beta(M + 0.5, 0.5)
    )
    sin_term = np.sin(np.pi * x) ** (2 * M - i)
    cos_term = np.cos(np.pi * x) ** (i)
    return cst * sin_term * cos_term


def transition_matrix(M: int = 10) -> ArrayLike:
    rank = 2 * M + 1
    P = np.zeros((rank, rank))
    for i, j in tqdm(
        np.ndindex((rank, rank)), total=rank**2, desc="Computing transition matrix"
    ):
        P[i, j] = scipy.integrate.quad(
            lambda x: noise_features(x, i, M=M)
            * noise_features(logistic_map(x), j, M=M),
            0,
            1,
        )[0]
    return P


class TrigonometricNoise:
    def __init__(self, M: int):
        self.M = M
        self.norm = np.pi / scipy.special.beta(M + 0.5, 0.5)

    def pdf(self, x):
        return self.norm * ((np.cos(np.pi * x)) ** (2 * self.M))


def make_noise_rng(M: int, rng_seed: int | None = None):
    random_state = np.random.default_rng(rng_seed)
    noise_dist = TrigonometricNoise(M)
    noise_rng = NumericalInversePolynomial(
        noise_dist,  # ty: ignore
        domain=(-0.5, 0.5),
        mode=0,
        random_state=random_state,
    )
    return noise_rng


def step(x: ArrayLike, noise_rng: NumericalInversePolynomial):
    x = np.asarray(x)
    y = logistic_map(x)
    xi = noise_rng.rvs(x.shape)
    x_next = np.mod(y + xi, 1)
    return x_next


def invariant_distribution(
    M: int = 10, precomputed_transition_matrix: ArrayLike | None = None
) -> callable:
    if precomputed_transition_matrix is None:
        P = transition_matrix(M=M)
    else:
        P = precomputed_transition_matrix
    values, vectors = np.linalg.eig(P.T)
    assert np.allclose(values[0], 1.0)
    mesh_size = 2**10
    x = np.linspace(0, 1, mesh_size + 1)
    dx = x[1] - x[0]
    pi = np.zeros_like(x)
    for basis_idx, coeff in enumerate(vectors[:, 0]):
        basis = noise_features(x, basis_idx, M=M)

        assert not np.iscomplex(coeff)
        pi += basis * coeff.real
    mass = scipy.integrate.romb(pi, dx)
    pi /= mass
    return lambda _x: np.interp(_x, x, pi)


def equilibrium_density_ratio(
    x,
    y,
    M: int = 10,
    precomputed_transition_matrix: ArrayLike | None = None,
    return_invariant_distribution: bool = False,
):
    pi = invariant_distribution(
        M=M, precomputed_transition_matrix=precomputed_transition_matrix
    )
    transition_fn = TrigonometricNoise(M)
    density_ratio = transition_fn.pdf(logistic_map(x) - y) / pi(y)
    if return_invariant_distribution:
        return density_ratio, pi
    else:
        return density_ratio
