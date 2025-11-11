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


def _eval_eigenfunctions(eigenvectors, eval_points, M: int = 10):
    """
    Evaluate 1D eigenfunctions from cosine basis expansion coefficients.

    Args:
        eigenvectors (ndarray): (num_basis, num_modes) matrix of eigenvector coefficients
        eval_points (ndarray): Points where to evaluate the eigenfunctions

    Returns:
        eigenfunctions (ndarray): Shape (len(eval_points), num_modes)
                                  eigenfunction values at evaluation points
    """
    eigenfunctions = np.zeros(
        eval_points.shape + (eigenvectors.shape[1],), dtype=eigenvectors.dtype
    )
    for i, coeffs in enumerate(eigenvectors):
        u_basis = noise_features(logistic_map(eval_points), i, M=M)  # eval_points.shape
        eigenfunctions += u_basis[..., None] * coeffs

    return eigenfunctions


def compute_logistic_map_eig(M: int = 10, eval_right_on=None, num_components: int = -1):
    """
    Computes the eigenvalues and eigenfunctions of the Koopman operator for the
    logistic map with trigonometric noise.

    The Koopman operator for the noisy logistic map :math:`x_{t+1} = r x_t (1 - x_t) + \\xi_t`
    with :math:`r=4` and trigonometric noise :math:`\\xi_t` has known eigenfunctions
    that can be computed analytically using a basis of trigonometric functions.

    Parameters
    ----------
    M : int, default=10
        Order of the trigonometric noise distribution. This determines the
        number of basis functions used (2M + 1).
    eval_right_on : ndarray of shape (n_samples,), optional
        Data points on which to evaluate the **right** eigenfunctions.
        If ``None``, right eigenfunctions are not evaluated.
    num_components : int, default=-1
        Number of dominant eigenvalues and corresponding eigenfunctions to return.
        If -1, all available components are returned.

    Returns
    -------
    eigenvalues : ndarray
        The dominant Koopman eigenvalues, sorted by magnitude in ascending order.
        Shape ``(num_components,)``.
    eigenfunctions : ndarray
        The leading right eigenfunctions evaluated at `eval_right_on`` if not ``None``.
        Shape ``(len(eval_right_on), num_components)``.

    Notes
    -----
    The eigenfunctions are computed using the basis functions:

    .. math::

        \\phi_i(x) = c_i \\sin^{2M-i}(\\pi x) \\cos^i(\\pi x)

    for :math:`i = 0, 1, \\ldots, 2M`.
    """

    P = transition_matrix(M=M)
    values, vectors = np.linalg.eig(P)
    if num_components == -1:
        num_components = vectors.shape[1]
    if eval_right_on is not None:
        eigenfunctions = _eval_eigenfunctions(
            vectors[:, :num_components], eval_right_on, M=M
        )
        return values[:num_components], eigenfunctions
    else:
        return values[:num_components]


def compute_logistic_map_invariant_pdf(M: int = 10) -> callable:
    """
    Computes the invariant probability density function (PDF) for the logistic map
    with trigonometric noise.

    The invariant PDF :math:`\\pi(x)` satisfies the Frobenius-Perron equation:

    .. math::

        \\pi(y) = \\int p(y|x) \\pi(x) dx

    where :math:`p(y|x)` is the transition density. For the noisy logistic map
    with :math:`r=4` and trigonometric noise, the invariant PDF can be
    derived from the leading eigenfunction of the Koopman operator.

    Parameters
    ----------
    M : int, default=10
        Order of the trigonometric noise distribution. This determines the
        basis functions used for computing the transition matrix.

    Returns
    -------
    invariant_pdf : callable
        A function that takes a scalar or array-like `x` as input and returns
        the value of the invariant PDF at those points.

    """

    return invariant_distribution(M=M)
