from math import sqrt

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.special
from scipy.stats.sampling import NumericalInversePolynomial


def make_duffing(
    X0,
    n_steps=100,
    dt=0.01,
    alpha=0.5,
    beta=0.0625,
    gamma=0.1,
    delta=2.5,
    omega=2.0,
):
    """
    Generate a trajectory from the Duffing oscillator.

    The Duffing oscillator is a damped driven nonlinear harmonic oscillator
    commonly used to study chaotic dynamics in physics and engineering.

    The system is governed by the differential equation:

    .. math::

        x'' + \\delta x' + \\alpha x + \\beta x^3 = \\gamma \\text{cos}(\\omega t).

    Parameters
    ----------
    X0 : array-like, shape (2,)
        Initial conditions [position, velocity].

    n_steps : int, default=100
        Number of time steps to simulate.

    dt : float, default=0.01
        Time step size for numerical integration.

    alpha : float, default=0.5
        Linear stiffness coefficient.

    beta : float, default=0.0625
        Nonlinear stiffness coefficient (cubic term).

    gamma : float, default=0.1
        Amplitude of the driving force.

    delta : float, default=2.5
        Damping coefficient.

    omega : float, default=2.0
        Angular frequency of the driving force.

    Returns
    -------
    df : pandas.DataFrame
        Trajectory of the oscillator with columns ``['position', 'velocity']``
        and ``n_steps + 1`` samples. Has a MultiIndex with levels ``['step', 'time']``.
        Metadata stored in ``df.attrs`` includes:

        - ``'generator'``: ``'make_duffing'``;
        - ``'X0'``: initial conditions;
        - ``'params'``: dict of all parameters.

    Examples
    --------
    >>> import numpy as np
    >>> X0 = np.array([0.0, 0.0])
    >>> df = make_duffing(X0, n_steps=1000, dt=0.01)
    >>> df.shape
    (1001, 2)
    >>> df.columns.tolist()
    ['position', 'velocity']
    >>> df.attrs['generator']
    'make_duffing'

    >>> # Access time values from index
    >>> times = df.index.get_level_values('time')

    >>> # Generate chaotic trajectory
    >>> df_chaotic = make_duffing(
    ...     X0=[0.1, 0.1],
    ...     n_steps=5000,
    ...     alpha=-1.0,
    ...     beta=1.0,
    ...     gamma=0.3,
    ...     delta=0.25,
    ...     omega=1.0
    ... )

    """
    X0 = np.asarray(X0)
    if X0.shape != (2,):
        raise ValueError(f"X0 must have shape (2,), got {X0.shape}")

    def duffing_rhs(t, x):
        """Right-hand side of the Duffing oscillator ODE."""
        return np.array(
            [
                x[1],
                -delta * x[1]
                - alpha * x[0]
                - beta * x[0] ** 3
                + gamma * np.cos(omega * t),
            ]
        )

    t_eval = np.arange(0, (n_steps + 1) * dt, dt)
    t_span = (0, t_eval[-1])

    sol = scipy.integrate.solve_ivp(
        duffing_rhs, t_span, X0, t_eval=t_eval, method="RK45"
    )

    # Create MultiIndex: (step, time)
    step_index = np.arange(len(sol.t))
    index = pd.MultiIndex.from_arrays([step_index, sol.t], names=["step", "time"])

    # Create DataFrame
    df = pd.DataFrame(sol.y.T, columns=["position", "velocity"], index=index)

    # Store metadata
    df.attrs = {
        "generator": "make_duffing",
        "X0": X0.tolist(),
        "params": {
            "n_steps": n_steps,
            "dt": dt,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "omega": omega,
        },
    }

    return df


def make_lorenz63(
    X0,
    n_steps=100,
    dt=0.01,
    sigma=10.0,
    mu=28.0,
    beta=2.667,  # 8/3
):
    """
    Generate a trajectory from the Lorenz-63 system.

    The Lorenz-63 system is a simplified mathematical model of atmospheric
    convection that exhibits chaotic behavior. It is one of the most famous
    examples of deterministic chaos.

    The system is governed by:

    .. math::

        \\begin{cases}
        \\frac{dx}{dt} = \\sigma(y - x) \\\\
        \\frac{dy}{dt} = x(\\mu - z) - y \\\\
        \\frac{dz}{dt} = xy - \\beta z
        \\end{cases}

    Parameters
    ----------
    X0 : array-like, shape (3,)
        Initial conditions ``[x, y, z]``.

    n_steps : int, default=100
        Number of time steps to simulate.

    dt : float, default=0.01
        Time step size for numerical integration.

    sigma : float, default=10.0
        The :math:`\\sigma` parameter, controlling the ratio of the rate of
        heat conduction to the rate of convection.

    mu : float, default=28.0
        The :math:`\\mu` parameter (also called :math:`\\rho`), representing
        the Rayleigh number.

    beta : float, default=8/3
        The :math:`\\beta` parameter, related to the physical dimensions of
        the convection layer.

    Returns
    -------
    df : pandas.DataFrame
        Trajectory of the Lorenz-63 system with columns ``['x', 'y', 'z']`` and
        ``n_steps + 1`` samples.
        Has a MultiIndex with levels ``['step', 'time']``.
        Metadata stored in ``df.attrs`` includes:

        - ``'generator'``: ``'make_lorenz63'``;
        - ``'X0'``: initial conditions;
        - ``'params'``: dict of all parameters.

    Examples
    --------
    >>> import numpy as np
    >>> X0 = np.array([1.0, 0.0, 0.0])
    >>> df = make_lorenz63(X0, n_steps=1000, dt=0.01)
    >>> df.shape
    (1001, 3)
    >>> df.columns.tolist()
    ['x', 'y', 'z']
    >>> df.attrs['generator']
    'make_lorenz63'

    Access time values from index:

    >>> times = df.index.get_level_values('time')

    Generate classic chaotic trajectory from near the attractor:

    >>> df_chaotic = make_lorenz63(
    ...     X0=[0.0, 1.0, 1.05],
    ...     n_steps=10000,
    ...     dt=0.01,
    ...     sigma=10.0,
    ...     mu=28.0,
    ...     beta=8.0/3.0
    ... )

    """
    X0 = np.asarray(X0)
    if X0.shape != (3,):
        raise ValueError(f"X0 must have shape (3,), got {X0.shape}")

    # Precompute linearized matrix for efficiency
    M_lin = np.array([[-sigma, sigma, 0], [mu, 0, 0], [0, 0, -beta]])

    def lorenz63_rhs(t, x):
        """Right-hand side of the Lorenz-63 system."""
        dx = M_lin @ x
        dx[1] -= x[2] * x[0]
        dx[2] += x[0] * x[1]
        return dx

    t_eval = np.arange(0, (n_steps + 1) * dt, dt)
    t_span = (0, t_eval[-1])

    sol = scipy.integrate.solve_ivp(
        lorenz63_rhs, t_span, X0, t_eval=t_eval, method="RK45"
    )

    # Create MultiIndex: (step, time)
    step_index = np.arange(len(sol.t))
    index = pd.MultiIndex.from_arrays([step_index, sol.t], names=["step", "time"])

    # Create DataFrame
    df = pd.DataFrame(sol.y.T, columns=["x", "y", "z"], index=index)

    # Store metadata
    df.attrs = {
        "generator": "make_lorenz63",
        "X0": X0.tolist(),
        "params": {
            "n_steps": n_steps,
            "dt": dt,
            "sigma": sigma,
            "mu": mu,
            "beta": beta,
        },
    }

    return df


def make_linear_system(
    X0,
    A,
    n_steps=100,
    noise=0.0,
    dt=1.0,
    random_state=None,
):
    """
    Generate a trajectory from a discrete-time linear dynamical system.

    The linear dynamical system is governed by:

    .. math::

        x_{t+1} = A x_t + \\xi_t

    where :math:`\\xi_t \\sim \\mathcal{N}(0, \\sigma^2 I)` is zero-mean
    Gaussian noise with standard deviation :math:`\\sigma`.

    For this linear system, the Koopman operator is simply the transpose
    of :math:`A`.

    Parameters
    ----------
    X0 : array-like, shape (d,)
        Initial conditions, where ``d`` is the system dimension.

    A : array-like, shape (d, d)
        State transition matrix defining the linear dynamics.

    n_steps : int, default=100
        Number of time steps to simulate.

    noise : float, default=0.0
        Standard deviation of the zero-mean Gaussian noise :math:`\\sigma`.

    dt : float, default=1.0
        Time step size. For discrete-time systems, this is typically 1.0.

    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for the noise.
        Pass an ``int`` for reproducible output across multiple function calls.

    Returns
    -------
    df : pandas.DataFrame
        Trajectory of the linear system with columns ``['x0', 'x1', ..., 'x{d-1}']``
        and ``n_steps + 1`` samples.
        Has a MultiIndex with levels ``['step', 'time']``.
        Metadata stored in ``df.attrs`` includes:

        - ``'generator'``: ``'make_linear_system'``;
        - ``'X0'``: initial conditions;
        - ``'A'``: state transition matrix;
        - ``'params'``: dict of all parameters.

    Examples
    --------
    >>> import numpy as np

    Stable 2D linear system:

    >>> A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    >>> X0 = np.array([1.0, 0.0])
    >>> df = make_linear_system(X0, A, n_steps=100)
    >>> df.shape
    (101, 2)
    >>> df.columns.tolist()
    ['x0', 'x1']

    With Gaussian noise:

    >>> df_noisy = make_linear_system(
    ...     X0=[1.0, 0.0],
    ...     A=A,
    ...     n_steps=100,
    ...     noise=0.1,
    ...     random_state=42
    ... )

    Unstable system (eigenvalues > 1):

    >>> A_unstable = np.array([[1.1, 0.0], [0.0, 1.1]])
    >>> df_unstable = make_linear_system(
    ...     X0=[0.1, 0.1],
    ...     A=A_unstable,
    ...     n_steps=50
    ... )

    Rotation matrix (periodic dynamics):

    >>> theta = np.pi / 4
    >>> A_rotation = np.array([
    ...     [np.cos(theta), -np.sin(theta)],
    ...     [np.sin(theta), np.cos(theta)]
    ... ])
    >>> df_rotation = make_linear_system(
    ...     X0=[1.0, 0.0],
    ...     A=A_rotation,
    ...     n_steps=100
    ... )

    Access the Koopman operator (transpose of A):

    >>> K = np.array(df.attrs['A']).T

    Notes
    -----
    The Koopman operator for this linear system is the transpose of the
    state transition matrix :math:`A`. This makes linear systems ideal
    test cases for Koopman operator learning algorithms.


    """
    X0 = np.asarray(X0)
    A = np.asarray(A)

    if X0.ndim != 1:
        raise ValueError(f"X0 must be 1-dimensional, got shape {X0.shape}")

    if A.ndim != 2:
        raise ValueError(f"A must be 2-dimensional, got shape {A.shape}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")

    d = A.shape[0]
    if X0.shape[0] != d:
        raise ValueError(f"X0 dimension ({X0.shape[0]}) must match A dimension ({d})")

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    # Preallocate trajectory array
    X = np.zeros((n_steps + 1, d))
    X[0] = X0

    # Simulate trajectory
    for i in range(n_steps):
        X[i + 1] = A @ X[i] + noise * rng.standard_normal(size=d)

    # Create time array
    t = np.arange(n_steps + 1) * dt

    # Create MultiIndex: (step, time)
    step_index = np.arange(n_steps + 1)
    index = pd.MultiIndex.from_arrays([step_index, t], names=["step", "time"])

    # Create DataFrame
    columns = [f"x{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=columns, index=index)

    # Store metadata
    df.attrs = {
        "generator": "make_linear_system",
        "X0": X0.tolist(),
        "A": A.tolist(),
        "params": {
            "n_steps": n_steps,
            "noise": noise,
            "dt": dt,
            "random_state": random_state,
        },
    }

    return df


def make_logistic_map(
    X0,
    n_steps=100,
    r=4.0,
    M=10,
    dt=1.0,
    random_state=None,
):
    """
    Generate a trajectory from the logistic map with optional trigonometric noise :cite:t:`make_logistic_map-ostruszka2000dynamical`.

    The logistic map is a discrete-time dynamical system defined by:

    .. math::

        x_{t+1} = r x_t (1 - x_t) + \\xi_t

    where :math:`\\xi_t` is drawn from a trigonometric noise distribution
    when ``M > 0``.

    The classic chaotic logistic map uses :math:`r = 4`. For this system
    with trigonometric noise, the eigenfunctions of the Koopman operator
    can be computed analytically using the basis:

    .. math::

        \\phi_i(x) = c_i \\sin^{2M-i}(\\pi x) \\cos^i(\\pi x)

    for :math:`i = 0, 1, \\ldots, 2M`.

    Parameters
    ----------
    X0 : float or array-like, shape (1,)
        Initial condition. Must be in ``[0, 1]`` for standard logistic map.

    n_steps : int, default=100
        Number of time steps to simulate.

    r : float, default=4.0
        Growth rate parameter. The classic chaotic regime is at ``r = 4``.

    M : int, default=10
        Order of the trigonometric noise distribution. Higher ``M`` makes
        the noise distribution more peaked around zero. If ``M <= 0``, no
        noise is added.

    noise : float, default=0.0

    dt : float, default=1.0
        Time step size. For discrete-time systems, this is typically 1.0.

    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for the noise.
        Pass an ``int`` for reproducible output across multiple function calls.

    Returns
    -------
    df : pandas.DataFrame
        Trajectory of the logistic map with column ``['x']`` and ``n_steps + 1`` samples.
        Has a MultiIndex with levels ``['step', 'time']``.
        Metadata stored in ``df.attrs`` includes:

        - ``'generator'``: ``'make_logistic_map'``;
        - ``'X0'``: initial condition;
        - ``'params'``: dict of all parameters.

    Examples
    --------
    >>> import numpy as np

    Classic chaotic logistic map:

    >>> df = make_logistic_map(X0=0.1, n_steps=100, r=4.0)
    >>> df.shape
    (101, 1)
    >>> df.columns.tolist()
    ['x']

    With trigonometric noise:

    >>> df_noisy = make_logistic_map(
    ...     X0=0.1,
    ...     n_steps=1000,
    ...     r=4.0,
    ...     M=10,
    ...     random_state=42
    ... )

    Different growth rates:

    >>> # Period-2 orbit (r ≈ 3.2)
    >>> df_period2 = make_logistic_map(X0=0.5, n_steps=100, r=3.2)
    >>>
    >>> # Chaotic (r = 4.0)
    >>> df_chaotic = make_logistic_map(X0=0.5, n_steps=100, r=4.0)

    Access metadata:

    >>> df.attrs['params']['M']
    10

    Notes
    -----
    The trigonometric noise distribution has PDF:

    .. math::

        p(\\xi) = \\frac{\\pi}{B(M+0.5, 0.5)} \\cos^{2M}(\\pi \\xi)

    for :math:`\\xi \\in [-0.5, 0.5]`, where :math:`B` is the beta function.

    For the noisy logistic map with :math:`r = 4`, the Koopman operator
    has known eigenfunctions that can be computed using the companion
    function ``compute_logistic_map_eigenfunctions``.

    """
    # Handle X0
    X0 = np.asarray(X0).flatten()
    if X0.size != 1:
        raise ValueError(
            f"X0 must be scalar or 1D array with 1 element, got {X0.shape}"
        )
    X0 = float(X0[0])

    if not 0 <= X0 <= 1:
        raise ValueError(f"X0 must be in [0, 1] for standard logistic map, got {X0}")

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    # Setup noise generator if needed
    if M > 0:
        noise_rng = _make_noise_rng(M, rng)
    else:
        noise_rng = None

    # Initialize trajectory

    # Preallocate trajectory array
    X = np.zeros(n_steps + 1)
    X[0] = X0

    # Simulate trajectory
    for i in range(n_steps):
        # Logistic map
        X[i + 1] = r * X[i] * (1 - X[i])

        # Add noise if specified
        if noise_rng is not None:
            xi = noise_rng.rvs()
            X[i + 1] = np.clip(X[i + 1] + xi, 0, 1)

    # Create time array
    t = np.arange(n_steps + 1) * dt

    # Create MultiIndex: (step, time)
    step_index = np.arange(n_steps + 1)
    index = pd.MultiIndex.from_arrays([step_index, t], names=["step", "time"])

    # Create DataFrame
    df = pd.DataFrame(X, columns=["x"], index=index)

    # Store metadata
    df.attrs = {
        "generator": "make_logistic_map",
        "X0": X0,
        "params": {
            "n_steps": n_steps,
            "r": r,
            "M": M,
            "dt": dt,
            "random_state": random_state,
        },
    }

    return df


def _make_noise_rng(M, rng):
    """Create noise random number generator with trigonometric distribution."""

    class TrigonometricNoise:
        def __init__(self, M):
            self.M = M
            self.norm = np.pi / scipy.special.beta(M + 0.5, 0.5)

        def pdf(self, x):
            return self.norm * ((np.cos(np.pi * x)) ** (2 * self.M))

    noise_dist = TrigonometricNoise(M)
    noise_rng = NumericalInversePolynomial(
        noise_dist, domain=(-0.5, 0.5), mode=0, random_state=rng
    )
    return noise_rng


def make_regime_switching_var(
    X0,
    phi1,
    phi2,
    transition,
    n_steps=100,
    dt=1.0,
    noise=0.0,
    random_state=None,
):
    """
    Generate a trajectory from a regime-switching vector autoregressive (VAR) process.

    This model alternates between two linear dynamical regimes according to a
    Markov transition matrix. At each step, the system evolves according to one
    of two dynamics matrices ``phi1`` or ``phi2``, with optional Gaussian noise.

    Mathematically, the system evolves as:

    .. math::

        x_{t+1} = \\Phi_{s_t} x_t + \\epsilon_t, \\quad
        \\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2 I)

    where ``s_t`` is the active regime (0 or 1), evolving according to a
    2x2 Markov transition matrix ``P`` such that

    .. math::

        P_{ij} = \\mathbb{P}(s_{t+1} = j \\mid s_t = i).

    Parameters
    ----------
    X0 : array-like of shape (n_features,)
        Initial state of the system.

    phi1 : ndarray of shape (n_features, n_features)
        Dynamics matrix for regime 0.

    phi2 : ndarray of shape (n_features, n_features)
        Dynamics matrix for regime 1.

    transition : ndarray of shape (2, 2)
        Row-stochastic Markov transition matrix defining the probabilities
        of switching between regimes. ``transition[i, j]`` gives the
        probability of transitioning from regime ``i`` to ``j``.

    n_steps : int, default=100
        Number of discrete time steps to simulate.

    dt : float, default=1.0
        Time step size for discrete simulation. Added as metadata in the output.

    noise : float, default=0.0
        Standard deviation of Gaussian noise added at each step.

    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for the noise.
        Pass an ``int`` for reproducible output across multiple function calls.

    Returns
    -------
    df : pandas.DataFrame
        Generated trajectory with columns ``['x0', 'x1', ..., 'x{n_features-1}']``
        and ``n_steps + 1`` samples. Has a MultiIndex with levels ``['step', 'time']``.

        The following metadata are stored in ``df.attrs``:

        - ``'generator'``: ``'make_regime_switching_var'``;
        - ``'X0'``: initial state;
        - ``'params'``: dictionary with all model parameters;
        - ``'regimes'``: integer array of active regimes over time.

    Examples
    --------
    >>> import numpy as np
    >>> from kooplearn.datasets import make_regime_switching_var
    >>> phi1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> phi2 = np.array([[0.5, -0.2], [0.3, 0.7]])
    >>> transition = np.array([[0.95, 0.05], [0.1, 0.9]])
    >>> X0 = np.zeros(2)
    >>> df = make_regime_switching_var(X0, phi1, phi2, transition, n_steps=100, noise=0.01)
    """

    X0 = np.asarray(X0, dtype=float)
    n_features = X0.shape[0]

    # Validate shapes
    phi1 = np.asarray(phi1, dtype=float)
    phi2 = np.asarray(phi2, dtype=float)
    transition = np.asarray(transition, dtype=float)

    if phi1.shape != (n_features, n_features):
        raise ValueError(
            f"`phi1` must have shape {(n_features, n_features)}, got {phi1.shape}."
        )
    if phi2.shape != (n_features, n_features):
        raise ValueError(
            f"`phi2` must have shape {(n_features, n_features)}, got {phi2.shape}."
        )
    if transition.shape != (2, 2):
        raise ValueError("`transition` must be a 2x2 matrix.")
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("Rows of `transition` must sum to 1.")

    rng = np.random.default_rng(random_state)
    phi = [phi1, phi2]

    # Allocate trajectory and regime arrays
    data = np.zeros((n_steps + 1, n_features))
    regimes = np.zeros(n_steps + 1, dtype=int)
    data[0] = X0

    current_regime = 0

    for t in range(n_steps):
        noise_vec = noise * rng.standard_normal(size=n_features)
        X_next = phi[current_regime] @ data[t] + noise_vec
        data[t + 1] = X_next

        # Draw next regime
        current_regime = rng.choice([0, 1], p=transition[current_regime])
        regimes[t + 1] = current_regime

    # Time and step index
    step_index = np.arange(n_steps + 1)
    time_index = step_index * dt
    index = pd.MultiIndex.from_arrays([step_index, time_index], names=["step", "time"])

    columns = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns, index=index)

    # Metadata
    df.attrs = {
        "generator": "make_regime_switching_var",
        "X0": X0.tolist(),
        "params": {
            "phi1": phi1,
            "phi2": phi2,
            "transition": transition,
            "n_steps": n_steps,
            "dt": dt,
            "noise": noise,
            "random_state": random_state,
        },
        "regimes": regimes,
    }

    return df


def make_prinz_potential(
    X0,
    n_steps=10000,
    dt=1e-4,
    gamma=1.0,
    sigma=sqrt(2.0),
    random_state=None,
):
    """
    Generate a 1D Langevin trajectory for the "Prinz potential" :cite:t:`make_prinz_potential-Prinz2011`.

    This quadruple-well potential exhibits three metastable states separated by
    energy barriers. The dynamics follow the (discretized) overdamped Langevin equation:

    .. math::

        X_{t + 1} = X_t -\\frac{1}{\\gamma}\\nabla V{X_t}\\Delta t + \\frac{\\sigma}{\\gamma}\\sqrt{\\Delta t}\\xi_t,


    where :math:`\\xi_t` is a Gaussian white noise process with zero mean and unit variance,
    :math:`\\gamma` is the friction coefficient, and :math:`k_B T = \\frac{\\sigma^2}{2\\gamma}` determines the thermal energy scale.

    The potential is defined as:

    .. math::

        V(x) = 32 x^8 - 256 e^{-80 x^2} - 80 e^{-40 (x + 0.5)^2}
               - 128 e^{-80 (x - 0.5)^2}.

    Parameters
    ----------
    X0 : float or array-like of shape (1,)
        Initial position.

    n_steps : int, default=10000
        Number of discrete time steps to simulate.

    dt : float, default=1e-4
        Time step size for Euler–Maruyama integration.

    gamma : float, default=0.1
        Friction coefficient.

    sigma : float, default=:math:`\\sqrt{2}`
        Noise variance, corresponding to a thermal energy scale
        :math:`k_B T = \\frac{\\sigma^2}{2\\gamma}`.

    random_state : int, RandomState instance or None, default=None
        Controls the random number generation for the noise.
        Pass an ``int`` for reproducible output across multiple function calls.

    Returns
    -------
    df : pandas.DataFrame
        Trajectory of the particle with column ``['x']`` and ``n_steps + 1`` samples.
        Indexed by a MultiIndex with levels ``['step', 'time']``.

        Metadata stored in ``df.attrs`` includes:

        - ``'generator'``: ``'make_prinz_potential'``;
        - ``'X0'``: initial condition;
        - ``'params'``: dictionary of all parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from kooplearn.datasets import make_prinz_potential
    >>> df = make_prinz_potential(X0=0.0, n_steps=5000, dt=1e-4)
    """

    X0 = np.atleast_1d(np.asarray(X0, dtype=float))
    if X0.shape != (1,):
        raise ValueError(f"X0 must have shape (1,), got {X0.shape}")
    X0 = X0[0]

    rng = np.random.default_rng(random_state)
    inv_gamma = 1.0 / gamma

    def force_fn(x):
        """Force corresponding to the triple-well potential."""
        return -1.0 * (
            -128 * np.exp(-80 * ((-0.5 + x) ** 2)) * (-0.5 + x)
            - 512 * np.exp(-80 * (x**2)) * x
            + 32 * (x**7)
            - 160 * np.exp(-40 * ((0.5 + x) ** 2)) * (0.5 + x)
        )

    X = np.zeros(n_steps + 1)
    X[0] = X0
    sqrt_term = inv_gamma * sigma * np.sqrt(dt)

    for t in range(n_steps):
        F = force_fn(X[t])
        xi = rng.standard_normal()
        X[t + 1] = X[t] + inv_gamma * F * dt + sqrt_term * xi

    # MultiIndex (step, time)
    step_index = np.arange(n_steps + 1)
    time_index = step_index * dt
    index = pd.MultiIndex.from_arrays([step_index, time_index], names=["step", "time"])

    df = pd.DataFrame(X[:, None], columns=["x"], index=index)
    df.attrs = {
        "generator": "make_prinz_potential",
        "X0": X0.tolist(),
        "params": {
            "n_steps": n_steps,
            "dt": dt,
            "gamma": gamma,
            "sigma": sigma,
            "random_state": random_state,
        },
    }

    return df
