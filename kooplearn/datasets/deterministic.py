import numpy as np
import scipy.integrate

from kooplearn.datasets.misc import DataGenerator


class DuffingOscillator(DataGenerator):
    """
    A class for simulating the Duffing Oscillator.

    The Duffing Oscillator is a mathematical model used to describe a damped
    driven harmonic oscillator with nonlinear effects. It is commonly used
    in physics and engineering to study chaotic behavior.

    Args:
        alpha (float, optional): The stiffness coefficient (default is 0.5).
        beta (float, optional): The nonlinear coefficient (default is 0.0625).
        gamma (float, optional): The damping coefficient (default is 0.1).
        delta (float, optional): Another damping coefficient (default is 2.5).
        omega (float, optional): The angular frequency of the driving force (default is 2.0).
        dt (float, optional): The time step size for the numerical integration (default is 0.01).

    Attributes:
        alpha (float): The stiffness coefficient.
        beta (float): The nonlinear coefficient.
        gamma (float): The damping coefficient.
        delta (float): Another damping coefficient.
        omega (float): The angular frequency of the driving force.
        dt (float): The time step size for numerical integration.

    Examples:

    .. code-block:: python

        duffing = DuffingOscillator(alpha=0.5, beta=0.0625, gamma=0.1, delta=2.5, omega=2.0, dt=0.01)
        initial_conditions = np.array([0.0, 0.0])
        trajectory = duffing.sample(initial_conditions, T=100)

    """

    def __init__(
        self, alpha=0.5, beta=0.0625, gamma=0.1, delta=2.5, omega=2.0, dt=0.01
    ):
        """
        Initializes a DuffingOscillator object.

        Args:
            alpha (float, optional): The stiffness coefficient (default is 0.5).
            beta (float, optional): The nonlinear coefficient (default is 0.0625).
            gamma (float, optional): The damping coefficient (default is 0.1).
            delta (float, optional): Another damping coefficient (default is 2.5).
            omega (float, optional): The angular frequency of the driving force (default is 2.0).
            dt (float, optional): The time step size for the numerical integration (default is 0.01).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.dt = dt

    def D(self, t, x):
        """
        The derivative function representing the Duffing oscillator's equations of motion.

        Args:
            t (float): The current time.
            x (np.ndarray): An array representing the current state [position, velocity].

        Returns:
            np.ndarray: An array representing the derivatives of the state [velocity, acceleration].
        """
        dx = np.array(
            [
                x[1],
                -self.delta * x[1]
                - self.alpha * x[0]
                - self.beta * x[0] ** 3
                + self.gamma * np.cos(self.omega * t),
            ]
        )
        return dx

    def sample(self, X0: np.ndarray, T: int = 1):
        """
        Generate the trajectory of the Duffing oscillator.

        Args:
            X0 (np.ndarray): The initial conditions as an array [initial_position, initial_velocity].
            T (int, optional): The number of time steps (default is 1).

        Returns:
            np.ndarray: An array containing the trajectory of the oscillator with shape (T+1, 2),
                where each row represents [position, velocity] at a given time step.
        """
        sim_time = self.dt * (T + 1)
        t_eval = np.linspace(0, sim_time, T + 1, endpoint=True)
        t_span = (0, t_eval[-1])
        sol = scipy.integrate.solve_ivp(
            self.D, t_span, X0, t_eval=t_eval, method="RK45"
        )
        return sol.y.T


class Lorenz63(DataGenerator):
    """
    A class for simulating the Lorenz-63 chaotic dynamical system.

    The Lorenz-63 system is a simplified mathematical model of atmospheric
    convection that exhibits chaotic behavior.

    Args:
        sigma (float, optional): The :math:`\\sigma` parameter (default is 10).
        mu (float, optional): The :math:`\\mu` parameter (default is 28).
        beta (float, optional): The :math:`\\beta` parameter (default is 8/3).
        dt (float, optional): The time step size for numerical integration (default is 0.01).

    Attributes:
        sigma (float): The :math:`\\sigma` parameter.
        mu (float): The :math:`\\mu` parameter.
        beta (float): The :math:`\\beta` parameter.
        dt (float): The time step size for numerical integration.
        M_lin (np.ndarray): The linearized matrix of the Lorenz-63 system.

    Examples:

    .. code-block:: python

        lorenz = Lorenz63(sigma=10, mu=28, beta=8/3, dt=0.01)
        initial_conditions = np.array([1.0, 0.0, 0.0])
        trajectory = lorenz.sample(initial_conditions, T=100)

    """

    def __init__(self, sigma=10, mu=28, beta=8 / 3, dt=0.01):
        """
        Initializes a Lorenz63 object.

        Args:
            sigma (float, optional): The :math:`\\sigma` parameter (default is 10).
            mu (float, optional): The :math:`\\mu` parameter (default is 28).
            beta (float, optional): The :math:`\\beta` parameter (default is 8/3).
            dt (float, optional): The time step size for numerical integration (default is 0.01).
        """
        self.sigma = sigma
        self.mu = mu
        self.beta = beta
        self.dt = dt
        self.M_lin = np.array(
            [[-self.sigma, self.sigma, 0], [self.mu, 0, 0], [0, 0, -self.beta]]
        )

    def sample(self, X0: np.ndarray, T: int = 1):
        """
        Generate the trajectory of the Lorenz-63 system.

        Args:
            X0 (np.ndarray): The initial conditions as an array [x, y, z].
            T (int, optional): The number of time steps (default is 1).

        Returns:
            np.ndarray: An array containing the trajectory of the system with shape (T+1, 3),
                where each row represents [x, y, z] at a given time step.
        """
        sim_time = self.dt * (T + 1)
        t_eval = np.linspace(0, sim_time, T + 1, endpoint=True)
        t_span = (0, t_eval[-1])
        sol = scipy.integrate.solve_ivp(
            self.D, t_span, X0, t_eval=t_eval, method="RK45"
        )
        return sol.y.T

    def D(self, t, x):
        """
        The derivative function representing the Lorenz-63 equations.

        Args:
            t (float): The current time.
            x (np.ndarray): An array representing the current state [x, y, z].

        Returns:
            np.ndarray: An array representing the derivatives of the state [dx/dt, dy/dt, dz/dt].
        """
        dx = self.M_lin @ x
        dx[1] -= x[2] * x[0]
        dx[2] += x[0] * x[1]
        return dx
