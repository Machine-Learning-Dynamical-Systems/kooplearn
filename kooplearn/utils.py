import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint


def check_if_resume_experiment(ckpt_call: Optional[ModelCheckpoint] = None):
    """ Checks if an experiment should be resumed based on the existence of checkpoint files.

    This function checks if the last checkpoint file and the best checkpoint file exist.
    If the best checkpoint file exists and the last checkpoint file does not, the experiment was terminated.
    If both files exist, the experiment was not terminated.

    The names of the best and last checkpoint files are automatically queried from the ModelCheckpoint instance. Thus,
    if user defines specific names for the checkpoint files, the function will still work as expected.

    Args:
        ckpt_call (Optional[ModelCheckpoint]): The ModelCheckpoint callback instance used in the experiment.
            If None, the function will return False, None, None.

    Returns:
        tuple: A tuple containing three elements:
            - terminated (bool): True if the experiment was terminated, False otherwise.
            - ckpt_path (pathlib.Path): The path to the last checkpoint file.
            - best_path (pathlib.Path): The path to the best checkpoint file.
    """
    if ckpt_call is None:
        return False, None, None

    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    if best_path.exists():
        return True, None, best_path
    elif ckpt_path.exists():
        return False, ckpt_path, None
    else:
        return False, None, None


@dataclass
class ModesInfo:
    """
    Data structure to store the data and utility functions used for dynamic mode decomposition.

    By default this data structure assumes that the state variables `z_k ∈ R^l` are real-valued. Thus the returned modes
    will not contain both complex-conjugate mode instances, but rather only their resultant real parts.

    This class takes an eigenvalue λ_i and uses its polar representation to compute relevant information of the
    eigenspace dynamics. That is for λ_i:= r_i * exp(j*θ_i), the r_i is the modulus and θ_i is the angle. We can then
    compute the frequency of oscillation and the decay rate of the mode dynamics z_k+1^i = r_i * exp(j*θ_i) * z_k^i.

    - Frequency: f_i = θ_i / (2π * dt)
    - Decay Rate: δ_i = ln(r_i) / dt


    Attributes:
        dt (float): Time step between discrete frames z_k and z_k+1, such that in continuous time z_k := z_(k*dt)
        eigvals (np.ndarray): Vector of N eigenvalues, λ_i ∈ C : i ∈ [1,...,N], of the evolution operator T. Shape: (N,)
        eigvecs_r (np.ndarray): Right eigenvectors stacked in column form in a matrix of shape (l, N), where each
        eigenvector, v_i : v_i ∈ C^l, i ∈ [1,...,N], of the evolution operator: T v_i = λ_i v_i.
        state_eigenbasis (np.ndarray): Vector of inner products <v_i, z_k> ∈ C between the state/latent state and the
        mode eigenvectors. Shape: (..,context_window, N)
          These values represent the scale and angle of state in each of the N eigenspaces of T.
        linear_decoder (Optional[np.ndarray]): Linear decoder matrix to project the modes from the latent space to
        the observation space. Shape: (l, o)
    """
    dt: float
    eigvals: np.ndarray
    eigvecs_r: np.ndarray
    state_eigenbasis: np.ndarray
    linear_decoder: Optional[np.ndarray] = None
    sort_by: str = "modulus"

    def __post_init__(self):
        """Identifies real and complex conjugate pairs of eigenvectors, along with their associated dimensions

        Assuming `z_k ∈ R^l` is real-valued, we will not obtain l modes, considering that for any eigenvector v_i
        associated with a complex eigenvalue λ_i ∈ C will have corresponding eigenpair (v_i^*, λ_i^*).
        Sort and cluster the eigenvalues by magnitude and field (real, complex)
        """
        # Sort and cluster the eigenvalues by magnitude and field (real, complex) ======================================
        from kooplearn._src.utils import parse_cplx_eig
        real_eigs, cplx_eigs, real_eigs_indices, cplx_eigs_indices = parse_cplx_eig(self.eigvals)
        self._state_dim = self.state_eigenbasis.shape[-1]

        if self.sort_by == "modulus":
            real_eigs_modulus = np.abs(real_eigs)
            cplx_eigs_modulus = np.abs(cplx_eigs)
            eigs_sort_metric = np.concatenate((real_eigs_modulus, cplx_eigs_modulus))
            eigs_indices = np.concatenate((real_eigs_indices, cplx_eigs_indices))
            # Sort the eigenvalues by modulus |λ_i| in descending order.
            sorted_indices = np.flip(np.argsort(eigs_sort_metric))
        else:
            raise NotImplementedError(f"Sorting by {self.sort_by} is not implemented yet.")

        # Store the sorted indices of eigenspaces by modulus, and the indices of real and complex eigenspaces.
        # Such that self.eigvals[self._sorted_eigs_indices] returns the sorted eigenvalues.
        self._sorted_eigs_indices = eigs_indices[sorted_indices]
        # Check resultant number of modes is equivalent to n_real_eigvals + 1/2 n_complex_eigvals
        assert len(self._sorted_eigs_indices) == len(cplx_eigs) + len(real_eigs)

        # Modify the eigenvalues, eigenvectors, and state_eigenbasis to be sorted, and ignore complex conjugates. =====
        self.eigvals = self.eigvals[self._sorted_eigs_indices]
        self.eigvecs_r = self.eigvecs_r[:, self._sorted_eigs_indices]
        self.state_eigenbasis = self.state_eigenbasis[..., self._sorted_eigs_indices]
        # Utility array to identify if in the new order the modes/eigvals are to be treated as complex or real
        self._is_complex_mode = [idx in cplx_eigs_indices for idx in self._sorted_eigs_indices]

        # Compute the real-valued modes associated with the values in `state_eigenbasis` ===============================
        # Change from the spectral/eigen-basis of the evolution operator to its original basis obtaining a tensor of
        # This process will generate N_u complex-valued mode vectors z_k^(i) ∈ C^l, where
        # N_u = n_real_eigvals + 1/2 n_complex_eigvals. The shape cplx_modes: (..., N_u, l)
        self.cplx_modes = np.einsum("...le,...e->...el", self.eigvecs_r, self.state_eigenbasis)
        if len(real_eigs) > 0:  # Check real modes have zero imaginary part
            _real_eigval_modes = self.cplx_modes[..., np.logical_not(self._is_complex_mode), :]
            assert np.allclose(_real_eigval_modes.imag, 0, rtol=1e-5, atol=1e-5), \
                f"Real modes have non-zero imaginary part: {np.max(_real_eigval_modes.imag)}"

    @property
    def n_modes(self):
        return len(self.eigvals)

    @property
    def modes(self):
        """ Compute the real-valued modes of the system.

        Each mode associated with a complex eigenvalue will be scaled to twice its real part, considering the conjugate
        pair v_i^* λ_i * <u_i,z_t> + v_i^* λ_i^* * <u_i^*,z_t>. This process will generate N_u complex-valued mode
        vectors z_k^(i) ∈ C^l, where N_u = n_real_eigvals + 1/2 n_complex_eigvals <= l.

        Returns:
            modes (np.ndarray): Array of shape (..., N_u, s) of real modes of the system, computed from the input
            `self.state_eigenbasis` of shape (..., l). Where s=l if `linear_decoder` is None, otherwise s=o.
            The modes are sorted by the selected metric in `sort_by`.
        """
        real_modes = self.cplx_modes.real
        real_modes[..., self._is_complex_mode, :] *= 2
        if self.linear_decoder is not None:
            real_modes = np.einsum("ol,...l->...o", self.linear_decoder, real_modes)
        return real_modes

    @property
    def modes_modulus(self):
        return np.abs(self.eigvals) ** (1 / self.dt)

    @property
    def modes_frequency(self):
        """Compute the frequency of oscilation of each mode's eigenspace dynamics

        Returns:
            np.ndarray: Array of frequencies in Hz of each mode's eigenspace dynamics. Shape: (N,)
        """
        angles = np.angle(self.eigvals)
        freqs = angles / (2 * np.pi * self.dt)
        return freqs

    @property
    def modes_amplitude(self):
        # Compute <u_i, z_k>.
        mode_amplitude = np.abs(self.state_eigenbasis)
        mode_amplitude[..., self._is_complex_mode] *= 2
        return mode_amplitude

    @property
    def modes_phase(self):
        mode_phase = np.angle(self.state_eigenbasis)
        # set phase of real modes to zero
        mode_phase[..., np.logical_not(self._is_complex_mode)] = 0
        return mode_phase

    @property
    def modes_decay_rate(self):
        """Compute the decay rate of each mode's eigenspace dynamics

        If the decay rate is positive, the mode initial condition will converge to 50% of its initial value at
        approximately 7 * decay_rate seconds.

        """
        decay_rates = np.log(np.abs(self.eigvals)) / self.dt
        return decay_rates

    @property
    def modes_transient_time(self):
        """

        Returns:
            Time in seconds until which the initial condition of the mode decays to 10% of its initial value
            (if decay rate is positive).
        """
        decay_rates = self.modes_decay_rate()
        return 1 / decay_rates * np.log(90. / 100.)
