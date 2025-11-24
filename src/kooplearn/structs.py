"""Structs used by the `kernel` algorithms."""

from dataclasses import dataclass
from typing import Iterator, Mapping, TypedDict

import numpy as np

from kooplearn._utils import find_complex_conjugates


@dataclass
class FitResult(Mapping[str, np.ndarray | None]):
    U: np.ndarray
    V: np.ndarray
    svals: np.ndarray | None = None

    def __post_init__(self):
        self.U = np.ascontiguousarray(self.U, dtype=np.float64)
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        if self.svals is not None:
            self.svals = np.ascontiguousarray(self.svals, dtype=np.float64)

    # --- Mapping interface -------------------------------------------------

    def __getitem__(self, key: str):
        if key == "U":
            return self.U
        if key == "V":
            return self.V
        if key == "svals":
            return self.svals
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("U", "V", "svals"))

    def __len__(self) -> int:
        return 3


class EigResult(TypedDict):
    """Return type for eigenvalue decompositions of kernel regressors."""

    values: np.ndarray
    left: np.ndarray | None
    right: np.ndarray


class PredictResult(TypedDict):
    """Return type for predictions of kernel regressors."""

    times: np.ndarray | None
    state: np.ndarray | None
    observable: np.ndarray | None


class DynamicalModes:
    """
    Container for dynamical modes from eigenvalue decomposition.

    This class stores and manages the modal decomposition of a dynamical system,
    including eigenvalues, eigenfunctions, and their projections. It automatically
    handles complex conjugate pairs, sorts modes by stability, and provides
    convenient access to mode shapes, frequencies, and decay rates.

    .. warning::

        The class should be not initialized directly, and will be the return type of ``.dynamical_modes`` methods of Kooplearn estimators.

    Parameters
    ----------
    values : np.ndarray, shape (rank,)
        1D array of eigenvalues (complex or real)
    right_eigenfunctions : np.ndarray, shape (n_points, rank)
        2D array of right eigenfunctions. Each column is an eigenfunction
        in the spatial domain.
    left_projections : np.ndarray, shape (rank, n_features)
        2D array of left projection vectors. Each row is a projection
        vector in the feature space.

    Attributes
    ----------
    n_modes : int
        Number of modes after filtering complex conjugate pairs

    Notes
    -----
    Complex conjugate pairs are automatically detected and only one from each
    pair is stored. When reconstructing modes from complex conjugate pairs,
    the real part is doubled to account for the missing conjugate:

    .. math::

        \\text{mode} = 2 \\cdot \\text{Re}(\\phi_r(x) \\langle \\phi_l, f \\rangle)

    where :math:`\\phi_r` is the right eigenfunction and :math:`\\langle \\phi_l, f \\rangle` is
    the left projection on the mode's observable.

    .. tip::

        Modes are sorted by stability: stable modes (:math:`|\\lambda| < 1`) are ordered by decreasing half-life, followed by unstable modes.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from kooplearn.datasets import make_duffing
        >>> from kooplearn.kernel import KernelRidge
        >>>
        >>> # Sample data from the Duffing oscillator
        >>> data = make_duffing(X0 = np.array([0, 0]), n_steps=1000)
        >>> data = data.to_numpy()
        >>>
        >>> # Fit the model
        >>> model = KernelRidge(n_components=4, kernel='rbf', alpha=1e-6, random_state=42)
        >>> model = model.fit(data)
        >>>
        >>> # Initialize the container
        >>> modes = model.dynamical_modes(data)
        >>>
        >>> # Access individual mode
        >>> mode_0 = modes[0]  # Returns (1001, 2) real array
        >>> print(f"Mode shape: {mode_0.shape}")
        Mode shape: (1001, 2)
        >>>
        >>> # Iterate over all modes
        >>> for idx, mode in enumerate(modes):
        ...     print(f"Mode {idx}: shape={mode.shape}, frequency={modes.frequency(idx):.3f}")
        Mode 0: shape=(1001, 2), frequency=0.000
        Mode 1: shape=(1001, 2), frequency=0.003
        Mode 2: shape=(1001, 2), frequency=0.000
        >>>
        >>> # Get summary statistics
        >>> summary_df = modes.summary(dt=0.1)
        >>> # Filter and analyze stable modes
        >>> stable_modes = summary_df[summary_df['is_stable']]
        >>> print(f"Number of stable modes: {len(stable_modes)}")
        Number of stable modes: 3
        >>>
        >>> slowest_decay = stable_modes.loc[stable_modes['lifetime'].idxmax()]
        >>> print(f"Slowest decay: lifetime={slowest_decay['lifetime']:.1f}s")
        Slowest decay: lifetime=69258.2s
    """

    def __init__(
        self,
        values: np.ndarray[np.complexfloating],
        right_eigenfunctions: np.ndarray[np.complexfloating],
        left_projections: np.ndarray[np.complexfloating],
    ) -> None:
        # Validate inputs
        self._validate_inputs(values, right_eigenfunctions, left_projections)

        # Process and store the modal decomposition
        self._process_modes(values, right_eigenfunctions, left_projections)

    def _validate_inputs(
        self,
        values: np.ndarray,
        right_eigenfunctions: np.ndarray,
        left_projections: np.ndarray,
    ) -> None:
        """
        Validate input arrays for correct dimensions and shapes.

        Parameters
        ----------
        values : np.ndarray
            Eigenvalues array
        right_eigenfunctions : np.ndarray
            Right eigenfunctions array
        left_projections : np.ndarray
            Left projections array

        Raises
        ------
        ValueError
            If arrays have incorrect dimensions or incompatible shapes
        """
        if values.ndim != 1:
            raise ValueError(f"Eigenvalues must be 1D array, got shape {values.shape}")

        if right_eigenfunctions.ndim != 2:
            raise ValueError(
                f"Right eigenfunctions must be 2D array, got shape "
                f"{right_eigenfunctions.shape}"
            )

        if left_projections.ndim != 2:
            raise ValueError(
                f"Left projections must be 2D array, got shape {left_projections.shape}"
            )

        rank = values.shape[0]

        if left_projections.shape[0] != rank:
            raise ValueError(
                f"Left projections first dimension ({left_projections.shape[0]}) "
                f"must match number of eigenvalues ({rank})"
            )

        if right_eigenfunctions.shape[1] != rank:
            raise ValueError(
                f"Right eigenfunctions second dimension "
                f"({right_eigenfunctions.shape[1]}) must match number of "
                f"eigenvalues ({rank})"
            )

    def _process_modes(
        self,
        values: np.ndarray,
        right_eigenfunctions: np.ndarray,
        left_projections: np.ndarray,
    ) -> None:
        """
        Process modal decomposition: filter conjugate pairs, compute properties, sort.

        This method performs the following steps:

        1. Identify and filter complex conjugate pairs
        2. Extract only unique modes (one from each conjugate pair)
        3. Compute frequencies and lifetimes
        4. Sort modes by stability (stable modes first, ordered by lifetimes)

        Parameters
        ----------
        values : np.ndarray
            Eigenvalues
        right_eigenfunctions : np.ndarray
            Right eigenfunctions
        left_projections : np.ndarray
            Left projections
        """
        # Step 1: Identify complex conjugate pairs
        # cc_pairs: array of shape (n_pairs, 2) with indices of conjugate pairs
        # real_idxs: array of indices for real eigenvalues
        cc_pairs, real_idxs = find_complex_conjugates(values)

        # Step 2: Select unique modes (first from each conjugate pair + all real)
        # For each conjugate pair [i, j], we only keep index i
        if len(cc_pairs) > 0:
            unique_indices = np.concatenate([cc_pairs[:, 0], real_idxs])
        else:
            unique_indices = real_idxs

        # Extract the unique eigenvalues and corresponding eigenfunctions/projections
        self._values = values[unique_indices]
        self._left_projections = left_projections[unique_indices, :]
        self._right_eigenfunctions = right_eigenfunctions[:, unique_indices]

        # Store complex conjugate pair information for mode reconstruction
        # Create a boolean mask: True if mode came from a conjugate pair
        self._cc_pair_mask = np.zeros(len(unique_indices), dtype=bool)
        n_cc_pairs = cc_pairs.shape[0]
        self._cc_pair_mask[:n_cc_pairs] = True

        # Step 3: Compute mode properties (frequency and lifetime)
        self._compute_mode_properties()

        # Step 4: Sort modes by stability
        self._sort_modes()

    def _compute_mode_properties(self) -> None:
        """
        Compute frequency and lifetime for each mode.

        Frequency is computed from the argument (phase angle) of the eigenvalue:

        .. math::

                f = \\frac{|\\arg(\\lambda)|}{2\\pi}

        Life-times (decay time constant) is computed from the magnitude:

        .. math::

                \\tau = \\begin{cases}
                -\\frac{1}{\\log|\\lambda|} & \\text{if } |\\lambda| < 1 \\\\
                \\infty & \\text{if } |\\lambda| \\geq 1
            \\end{cases}

        Notes
        -----
        The lifetime formula gives the e-folding time (time for amplitude to
        decay by a factor of e ≈ 2.718). To get the half-life (time to
        decay by factor of 2), multiply by ln(2) ≈ 0.693.

        The derivation follows from the discrete-time evolution:

        .. math::

            a_n = a_0 \\lambda^n \\implies |a_n| = |a_0| |\\lambda|^n

        Setting :math:`|a_n| = |a_0|/e` and solving for n gives :math:`\\tau = -1/\\log|\\lambda|`.
        """
        # Compute magnitude and phase of eigenvalues
        magnitude = np.abs(self._values)
        phase = np.angle(self._values)

        # Frequency from phase angle (oscillations per time step)
        # Take absolute value since we only store one of each conjugate pair
        self._frequencies = np.abs(phase) / (2.0 * np.pi)

        # Decay time constant (e-folding time) for stable modes
        # For |λ| < 1: amplitude decays as |λ|^n, so ln(amplitude) = n*ln(|λ|)
        # Time to decay by factor e: n = 1/|ln(|λ|)| = -1/ln(|λ|)
        self._lifetimes = np.where(
            magnitude < 1.0,
            -1.0 / np.log(magnitude),  # Decay time constant
            np.inf,  # Unstable modes don't decay
        )

    def _sort_modes(self) -> None:
        """
        Sort modes by stability: stable modes first (by decreasing half-life),
        then unstable modes.

        Sorting strategy:

        - Stable modes (:math:`|\\lambda| < 1`) are sorted by magnitude in descending order
          (modes closer to :math:`|\\lambda| = 1` have longer half-lives and appear first)
        - Unstable modes (:math:`|\\lambda| \\geq 1`) come after all stable modes

        Notes
        -----
        The sort key is constructed as:

        .. math::

            \\text{key} = \\begin{cases}
                |\\lambda| & \\text{if } |\\lambda| < 1 \\\\
                -\\infty & \\text{if } |\\lambda| \\geq 1
            \\end{cases}

        Sorting in descending order places stable modes first (largest :math:`|\\lambda|` first),
        followed by unstable modes.
        """
        magnitude = np.abs(self._values)

        # Create sort key: stable modes get their magnitude, unstable get -inf
        # This puts unstable modes at the end after sorting in descending order
        sort_key = np.where(magnitude < 1.0, magnitude, -np.inf)

        # Sort in descending order (most stable first)
        sort_indices = np.argsort(sort_key)[::-1]

        # Apply sorting to all stored arrays
        self._values = self._values[sort_indices]
        self._left_projections = self._left_projections[sort_indices, :]
        self._right_eigenfunctions = self._right_eigenfunctions[:, sort_indices]
        self._frequencies = self._frequencies[sort_indices]
        self._lifetimes = self._lifetimes[sort_indices]
        self._cc_pair_mask = self._cc_pair_mask[sort_indices]

    def _validate_index(self, key: int) -> None:
        """
        Validate that an index is within valid range.

        Parameters
        ----------
        key : int
            Index to validate

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range
        """
        if not isinstance(key, int):
            raise TypeError(f"Index must be an integer, got {type(key).__name__}")

        if key < 0 or key >= self.n_modes:
            raise IndexError(
                f"Index {key} is out of range for container with {self.n_modes} modes"
            )

    @property
    def n_modes(self) -> int:
        """
        Number of modes in the container.

        Returns
        -------
        int
            Total number of modes after filtering complex conjugate pairs
        """
        return self._values.shape[0]

    def __len__(self) -> int:
        """
        Return the number of modes.

        Returns
        -------
        int
            Number of modes in the container
        """
        return self.n_modes

    def __getitem__(self, key: int) -> np.ndarray:
        """
        Get the spatial mode shape at the given index.

        The mode is reconstructed as the outer product of the right eigenfunction
        and left projection:

        .. math::

            \\text{mode}_{ij} = \\phi_r[i] \\cdot \\phi_l[j]

        For complex conjugate pairs, the real part is doubled:

        .. math::

            \\text{mode} = 2 \\cdot \\text{Re}(\\phi_r \\otimes \\phi_l^*)

        This accounts for the contribution of both conjugates since
        :math:`z + z^* = 2\\text{Re}(z)`.

        Parameters
        ----------
        key : int
            Index of the mode to retrieve (0 to n_modes-1)

        Returns
        -------
        mode : np.ndarray, shape (n_points, n_features)
            2D real array containing the spatial mode shape

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range
        """
        self._validate_index(key)

        # Extract eigenfunction and projection for this mode
        right_vector = self._right_eigenfunctions[:, key]  # Shape: (n_points,)
        left_vector = self._left_projections[key, :]  # Shape: (n_features,)

        # Compute outer product: mode[i,j] = right[i] * left[j]
        # Using np.outer for clarity and efficiency
        mode = np.outer(right_vector, left_vector).real

        # For complex conjugate pairs, double the real part
        # This accounts for the contribution of both conjugates: z + z* = 2*Re(z)
        if self._cc_pair_mask[key]:
            mode *= 2.0

        return mode

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over all modes in the container.

        Yields
        ------
        mode : np.ndarray, shape (n_points, n_features)
            Mode shape arrays in order
        """
        for i in range(self.n_modes):
            yield self[i]

    def frequency(self, key: int, dt: float = 1.0) -> float:
        """
        Get the oscillation frequency of a mode in physical time units.

        Parameters
        ----------
        key : int
            Index of the mode
        dt : float, optional
            Time step size, by default 1.0. Used to convert from per-timestep
            to per-unit-time frequencies: :math:`f_{\\text{physical}} = f_{\\text{discrete}} / \\Delta t`

        Returns
        -------
        float
            Frequency in cycles per unit time

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range

        Notes
        -----
        The returned frequency is in cycles per unit time (Hz if time is in seconds).
        For angular frequency (rad/time), multiply by :math:`2\\pi`.

        .. math::

            \\omega = 2\\pi f
        """
        self._validate_index(key)
        return self._frequencies[key] / dt

    def lifetime(self, key: int, dt: float = 1.0) -> float:
        """
        Get the decay time constant (e-folding time) of a mode.

        Parameters
        ----------
        key : int
            Index of the mode
        dt : float, optional
            Time step size, by default 1.0. Used to convert from timesteps
            to physical time units: :math:`\\tau_{\\text{physical}} = \\tau_{\\text{discrete}} \\times \\Delta t`

        Returns
        -------
        float
            Time constant in physical time units. Returns ``np.inf`` for unstable modes.

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range

        Notes
        -----
        This returns the e-folding time (time for amplitude to decay by factor e ≈ 2.718).
        For the actual half-life (time to decay by half), multiply by ln(2) ≈ 0.693:

        .. math::

            t_{1/2} = \\tau \\cdot \\ln(2)
        """
        self._validate_index(key)
        return self._lifetimes[key] * dt

    def summary(self, dt: float = 1.0):
        """
        Generate a summary DataFrame of all mode properties.

        Parameters
        ----------
        dt : float, optional
            Time step size, by default 1.0, for converting to physical units

        Returns
        -------
        pandas.DataFrame
            DataFrame with the following columns:

            - ``frequency`` : Oscillation frequency (cycles per unit time)
            - ``lifetime`` : Decay time constant (time units)
            - ``eigenvalue_real`` : Real part of eigenvalue
            - ``eigenvalue_imag`` : Imaginary part of eigenvalue
            - ``eigenvalue_magnitude`` : Magnitude of eigenvalue
            - ``is_stable`` : Boolean, True if |λ| < 1
            - ``is_conjugate_pair`` : Boolean, True if mode comes from conjugate pair

        Notes
        -----
        Requires pandas to be installed.
        """
        import pandas as pd

        magnitude = np.abs(self._values)

        return pd.DataFrame(
            {
                "frequency": self._frequencies / dt,
                "lifetime": self._lifetimes * dt,
                "eigenvalue_real": self._values.real,
                "eigenvalue_imag": self._values.imag,
                "eigenvalue_magnitude": magnitude,
                "is_stable": magnitude < 1.0,
                "is_conjugate_pair": self._cc_pair_mask,
            }
        )

    def get_eigenvalue(self, key: int) -> complex:
        """
        Get the eigenvalue for a specific mode.

        Parameters
        ----------
        key : int
            Index of the mode

        Returns
        -------
        complex
            The eigenvalue with positive imaginary part for conjugate pairs

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range
        """
        self._validate_index(key)
        # Return eigenvalue with positive imaginary part
        val = self._values[key]
        return val.real + 1j * np.abs(val.imag)

    def get_right_eigenfunction(self, key: int) -> np.ndarray[np.complexfloating]:
        """
        Get the right eigenfunction associated to a specific mode.

        Parameters
        ----------
        key : int
            Index of the mode

        Returns
        -------
        complex : np.ndarray, shape (n_points,)
            The right eigenfunction at index ``key``

        Raises
        ------
        TypeError
            If key is not an integer
        IndexError
            If key is out of range
        """
        self._validate_index(key)
        # Return eigenvalue with positive imaginary part
        val = self._right_eigenfunctions[:, key]
        return val
