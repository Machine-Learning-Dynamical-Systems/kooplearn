"""
Koopman operator analysis for the noisy logistic map.

This module implements analytical tools for studying the noisy logistic map with trigonometirc noise.

Main API Functions
------------------
compute_logistic_map_eig : Compute eigenvalues and eigenfunctions of the Koopman operator
compute_logistic_map_invariant_pdf : Compute the invariant probability density function

Utility Functions
-----------------
logistic_map : The deterministic logistic map (r=4)
noise_features : Trigonometric basis functions
compute_transition_matrix : Precompute transition matrix for efficiency
make_noise_rng : Create a random number generator for the noise
step : Simulate one step of the noisy map
TrigonometricNoise : The noise distribution class

Notes
-----
For repeated calls with the same M value, precompute the transition matrix
using `compute_transition_matrix(M)` and pass it to the main functions to
avoid redundant computation.
"""

from collections.abc import Callable

import numpy as np
import scipy.integrate
import scipy.special
from scipy.stats.sampling import NumericalInversePolynomial


def logistic_map(x: np.ndarray) -> np.ndarray:
    """The logistic map with r=4.

    Parameters
    ----------
    x : np.ndarray
        A scalar or array of values in the domain [0, 1].

    Returns
    -------
    np.ndarray
        The result of applying the logistic map to `x`.
    """
    return 4 * x * (1 - x)


def noise_features(x: np.ndarray, i: int, M: int = 10) -> np.ndarray:
    """Trigonometric features for the noisy logistic map.

    These features are basis functions used to approximate the eigenfunctions
    of the Transfer operator for the logistic map with trigonometric noise.

    Parameters
    ----------
    x : np.ndarray
        A scalar or array of values in the domain [0, 1].
    i : int
        The index of the feature.
    M : int, optional
        Order of the trigonometric noise distribution. Default is 10.

    Returns
    -------
    np.ndarray
        The `i`-th trigonometric feature evaluated at `x`.
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    cst = np.sqrt(np.pi * scipy.special.binom(2 * M, i) / scipy.special.beta(M + 0.5, 0.5))
    sin_term = np.sin(np.pi * x) ** (2 * M - i)
    cos_term = np.cos(np.pi * x) ** i
    return cst * sin_term * cos_term


def compute_transition_matrix(M: int = 10) -> np.ndarray:
    """Compute the transition matrix for the noisy logistic map.

    This matrix represents the action of the Transfer operator on the basis
    of trigonometric features. For large M values, this computation can be
    expensive. Precompute and cache this matrix if you need to call other
    functions multiple times with the same M.

    Parameters
    ----------
    M : int, optional
        Order of the trigonometric noise distribution. This determines the
        size of the transition matrix, which will be `(2*M + 1, 2*M + 1)`.
        Default is 10.

    Returns
    -------
    np.ndarray
        The transition matrix `P` of shape `(2*M + 1, 2*M + 1)`.

    Examples
    --------
    >>> # Precompute for efficiency
    >>> P = compute_transition_matrix(M=10)
    >>> eigenvalues = compute_logistic_map_eig(M=10, precomputed_transition_matrix=P)
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    rank = 2 * M + 1
    P = np.zeros((rank, rank))
    for i, j in np.ndindex((rank, rank)):
        P[i, j] = scipy.integrate.quad(
            lambda x: noise_features(x, i, M=M) * noise_features(logistic_map(x), j, M=M),
            0,
            1,
        )[0]
    return P


class TrigonometricNoise:
    """Trigonometric noise distribution.

    This class defines a probability distribution for trigonometric noise,
    which is used to perturb the logistic map. The PDF is proportional to
    :math:`\\cos(\\pi * x)^(2*M)`.

    Parameters
    ----------
    M : int
        Order of the trigonometric noise distribution. Must be positive.

    Attributes
    ----------
    M : int
        Order of the distribution.
    norm : float
        Normalization constant for the PDF.
    """

    def __init__(self, M: int):
        if M <= 0:
            raise ValueError(f"M must be positive, got {M}")
        self.M = M
        self.norm = np.pi / scipy.special.beta(M + 0.5, 0.5)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Parameters
        ----------
        x : np.ndarray
            A scalar or array of values.

        Returns
        -------
        np.ndarray
            The value of the PDF at `x`.
        """
        return self.norm * (np.cos(np.pi * x) ** (2 * self.M))


def make_noise_rng(M: int, random_state: int | None = None) -> NumericalInversePolynomial:
    """Create a random number generator for trigonometric noise.

    Parameters
    ----------
    M : int
        Order of the trigonometric noise distribution. Must be positive.
    random_state : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    NumericalInversePolynomial
        A random number generator for the trigonometric noise distribution.

    Examples
    --------
    >>> from kooplearn.datasets._logistic_map import make_noise_rng, step
    >>> rng = make_noise_rng(M=10, random_state=42)
    >>> x = np.array([0.5, 0.3, 0.7])
    >>> x_next = step(x, rng)
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    rng = np.random.default_rng(random_state)
    noise_dist = TrigonometricNoise(M)
    noise_rng = NumericalInversePolynomial(
        noise_dist,
        domain=(-0.5, 0.5),
        mode=0,
        random_state=rng,
    )
    return noise_rng


def step(x: np.ndarray, noise_rng: NumericalInversePolynomial) -> np.ndarray:
    """Perform one step of the noisy logistic map.

    Parameters
    ----------
    x : np.ndarray
        A scalar or array of values in the domain [0, 1].
    noise_rng : NumericalInversePolynomial
        A random number generator for the trigonometric noise.
        Create using `make_noise_rng()`.

    Returns
    -------
    np.ndarray
        The result of applying the noisy logistic map to `x`.

    Examples
    --------
    >>> from kooplearn.datasets._logistic_map import make_noise_rng, step
    >>> rng = make_noise_rng(M=10, random_state=42)
    >>> x = np.array([0.5])
    >>> x_next = step(x, rng)
    """
    x = np.asarray(x)
    y = logistic_map(x)
    xi = noise_rng.rvs(x.shape)
    x_next = np.mod(y + xi, 1)
    return x_next


def compute_invariant_distribution(M: int = 10, precomputed_transition_matrix: np.ndarray | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the invariant distribution of the noisy logistic map.

    This function computes the invariant distribution of the noisy logistic
    map by finding the leading eigenvector of the transition matrix.

    Parameters
    ----------
    M : int, optional
        Order of the trigonometric noise distribution. Default is 10.
    precomputed_transition_matrix : np.ndarray or None, optional
        A precomputed transition matrix from `compute_transition_matrix()`.
        If None, the matrix is computed. For repeated calls with the same M,
        precomputing the matrix significantly improves performance.
        Default is None.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that evaluates the invariant distribution at given points.

    Notes
    -----
    The invariant distribution π(x) satisfies the Frobenius-Perron equation
    and is computed from the leading (unit) eigenvector of P^T.
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    if precomputed_transition_matrix is None:
        P = compute_transition_matrix(M=M)
    else:
        P = precomputed_transition_matrix

    values, vectors = np.linalg.eig(P.T)

    # Find the eigenvalue closest to 1 (the leading eigenvalue)
    leading_idx = np.argmax(np.abs(values))
    if not np.allclose(values[leading_idx], 1.0):
        raise RuntimeError(f"Leading eigenvalue is {values[leading_idx]}, expected 1.0. The transition matrix may be incorrectly computed.")

    # Build the invariant distribution from basis functions
    mesh_size = 2**10
    x = np.linspace(0, 1, mesh_size + 1)
    dx = x[1] - x[0]
    pi = np.zeros_like(x)

    for basis_idx, coeff in enumerate(vectors[:, leading_idx]):
        basis = noise_features(x, basis_idx, M=M)
        pi += basis * coeff.real

    # Normalize to ensure it's a proper probability distribution
    mass = scipy.integrate.romb(pi, dx)
    pi /= mass

    return lambda _x: np.interp(_x, x, pi)


def _eval_eigenfunctions(eigenvectors: np.ndarray, eval_points: np.ndarray, M: int = 10) -> np.ndarray:
    """Evaluate eigenfunctions from basis expansion coefficients.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Array of shape `(num_basis, num_modes)` containing eigenvector coefficients.
    eval_points : np.ndarray
        Points where to evaluate the eigenfunctions.
    M : int, optional
        Order of the trigonometric noise distribution. Default is 10.

    Returns
    -------
    np.ndarray
        Array of shape `(*eval_points.shape, num_modes)` containing
        eigenfunction values at evaluation points.
    """
    eval_points = np.asarray(eval_points)
    eigenfunctions = np.zeros((*eval_points.shape, eigenvectors.shape[1]), dtype=eigenvectors.dtype)

    for i, coeffs in enumerate(eigenvectors):
        u_basis = noise_features(logistic_map(eval_points), i, M=M)
        eigenfunctions += u_basis[..., None] * coeffs

    return eigenfunctions


def compute_logistic_map_eig(
    M: int = 10,
    eval_right_on: np.ndarray | None = None,
    num_components: int = -1,
    precomputed_transition_matrix: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenfunctions of the Transfer operator.

    The Transfer operator for the noisy logistic map with trigonometric noise
    has eigenfunctions that can be computed
    analytically using a basis of trigonometric functions.

    Parameters
    ----------
    M : int, optional
        Order of the trigonometric noise distribution. This determines the
        number of basis functions used (2M + 1). Default is 10.
    eval_right_on : np.ndarray or None, optional
        Data points on which to evaluate the right eigenfunctions.
        If None, right eigenfunctions are not evaluated. Default is None.
    num_components : int, optional
        Number of dominant eigenvalues and corresponding eigenfunctions to return.
        If -1, all available components are returned. Default is -1.
    precomputed_transition_matrix : np.ndarray or None, optional
        A precomputed transition matrix from ``compute_transition_matrix()``.
        If None, the matrix is computed. For repeated calls with the same M,
        precomputing the matrix significantly improves performance.
        Default is None.

    Returns
    -------
    eigenvalues : np.ndarray
        The Koopman eigenvalues. Shape ``(num_components,)``.
    eigenfunctions : np.ndarray, optional
        The right eigenfunctions evaluated at `eval_right_on` if provided.
        Shape ``(len(eval_right_on), num_components)``.

    Notes
    -----
    The eigenfunctions are computed using the basis functions:

    .. math::

        \\phi_i(x) = c_i \\sin^{2M-i}(\\pi x) \\cos^i(\\pi x)

    for i = 0, 1, ..., 2M.

    Examples
    --------
    >>> # Compute eigenvalues only
    >>> eigenvalues = compute_logistic_map_eig(M=10)
    >>>
    >>> # Compute eigenvalues and eigenfunctions
    >>> x = np.linspace(0, 1, 100)
    >>> eigenvalues, eigenfunctions = compute_logistic_map_eig(M=10, eval_right_on=x)
    >>>
    >>> # Efficient repeated calls
    >>> P = compute_transition_matrix(M=10)
    >>> eig1 = compute_logistic_map_eig(M=10, precomputed_transition_matrix=P)
    >>> eig2 = compute_logistic_map_eig(M=10, precomputed_transition_matrix=P)
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    if precomputed_transition_matrix is None:
        P = compute_transition_matrix(M=M)
    else:
        P = precomputed_transition_matrix

    values, vectors = np.linalg.eig(P)

    if num_components == -1:
        num_components = vectors.shape[1]

    if eval_right_on is not None:
        eigenfunctions = _eval_eigenfunctions(vectors[:, :num_components], eval_right_on, M=M)
        return values[:num_components], eigenfunctions
    else:
        return values[:num_components]


def compute_logistic_map_invariant_pdf(M: int = 10, precomputed_transition_matrix: np.ndarray | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the invariant probability density function.

    The invariant PDF for the logistic map with trigonometric noise.
    This function is computed by solving the Frobenius-Perron equation:

    .. math::

        \\pi(y) = \\int p(y|x) \\pi(x) dx

    where p(y|x) is the transition density.

    Parameters
    ----------
    M : int, optional
        Order of the trigonometric noise distribution. This determines the
        basis functions used for computing the transition matrix. Default is 10.
    precomputed_transition_matrix : np.ndarray or None, optional
        A precomputed transition matrix from ``compute_transition_matrix()``.
        If None, the matrix is computed. For repeated calls with the large values of M,
        precomputing the matrix significantly improves performance.
        Default is None.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that takes a scalar or array-like `x` as input and returns
        the value of the invariant PDF at those points.

    Examples
    --------
    >>> pdf = compute_logistic_map_invariant_pdf(M=10)
    >>> x = np.linspace(0, 1, 100)
    >>> density = pdf(x)
    >>>
    >>> # Efficient repeated calls
    >>> P = compute_transition_matrix(M=10)
    >>> pdf1 = compute_logistic_map_invariant_pdf(M=10, precomputed_transition_matrix=P)
    >>> pdf2 = compute_logistic_map_invariant_pdf(M=10, precomputed_transition_matrix=P)
    """
    return compute_invariant_distribution(M=M, precomputed_transition_matrix=precomputed_transition_matrix)
