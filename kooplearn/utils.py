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
        elif self.sort_by == "modulus-amplitude":
            real_eigs_modulus = np.abs(real_eigs)
            cplx_eigs_modulus = np.abs(cplx_eigs)
            real_eigs_amplitude = np.abs(self.state_eigenbasis[0, 0, real_eigs_indices])
            cplx_eigs_amplitude = np.abs(self.state_eigenbasis[0, 0, cplx_eigs_indices])
            eigs_sort_metric = np.concatenate((real_eigs_modulus * real_eigs_amplitude,
                                               2 * cplx_eigs_modulus * cplx_eigs_amplitude))
            eigs_indices = np.concatenate((real_eigs_indices, cplx_eigs_indices))
            # Sort the eigenvalues by the product of modulus |λ_i| and amplitude |<v_i, z_k>| in descending order.
            sorted_indices = np.flip(np.argsort(eigs_sort_metric))
        elif self.sort_by == "freq":
            # Sort real-eigvals by modulus (decreasing), and complex eigvals by frequency (increasing)
            freqs_cplx = np.angle(cplx_eigs) / (2 * np.pi * self.dt)
            sorted_cplx_indices = cplx_eigs_indices[np.argsort(freqs_cplx)]
            sorted_real_indices = np.flip(real_eigs_indices[np.argsort(np.abs(real_eigs))])
            eigs_indices = np.arange(len(self.eigvals))
            sorted_indices = np.concatenate((sorted_real_indices, sorted_cplx_indices))
        else:
            raise NotImplementedError(f"Mode sorting by {self.sort_by} is not implemented yet.")

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
        self.is_complex_mode = [idx in cplx_eigs_indices for idx in self._sorted_eigs_indices]

        # If the input state_eigenbasis is a trajectory of states in time of shape (..., context_window, l),
        # we compute the predicted state_eigenbasis by applying the linear dynamics of each eigenspace to the
        # initial state_eigenbasis a.k.a the eigenfunctions evaluated at time 0.
        context_window = self.state_eigenbasis.shape[-2]
        eigfn_0 = self.state_eigenbasis[..., 0,:]
        eigval_t = np.asarray([self.eigvals ** t for t in range(context_window)])  # λ_i^t for t in [0,time_horizon)
        eigfn_pred = np.einsum("...l,tl->...tl", eigfn_0, eigval_t)  # (...,context_window, l)
        assert self.state_eigenbasis.shape == eigfn_pred.shape
        self.pred_state_eigenbasis = eigfn_pred

        # Compute the real-valued modes associated with the values in `state_eigenbasis` ===============================
        # Change from the spectral/eigen-basis of the evolution operator to its original basis obtaining a tensor of
        # This process will generate N_u complex-valued mode vectors z_k^(i) ∈ C^l, where
        # N_u = n_real_eigvals + 1/2 n_complex_eigvals. The shape cplx_modes: (..., N_u, l)
        self.cplx_modes = np.einsum("...le,...e->...el", self.eigvecs_r, self.state_eigenbasis)
        self.cplx_modes_pred = np.einsum("...le,...e->...el", self.eigvecs_r, self.pred_state_eigenbasis)
        if len(real_eigs) > 0:  # Check real modes have zero imaginary part
            _real_eigval_modes = self.cplx_modes[..., np.logical_not(self.is_complex_mode), :]
            assert np.allclose(_real_eigval_modes.imag, 0, rtol=1e-5, atol=1e-5), \
                f"Real modes have non-zero imaginary part: {np.max(_real_eigval_modes.imag)}"

    @property
    def n_modes(self):
        return len(self.eigvals)

    @property
    def n_dc_modes(self):
        return np.sum(np.logical_not(self.is_complex_mode))

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
        real_modes = self.cplx_modes_pred.real
        real_modes[..., self.is_complex_mode, :] *= 2
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
        mode_amplitude[..., self.is_complex_mode] *= 2
        return mode_amplitude

    @property
    def modes_phase(self):
        mode_phase = np.angle(self.state_eigenbasis)
        # set phase of real modes to zero
        mode_phase[..., np.logical_not(self.is_complex_mode)] = 0
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
        decay_rates = self.modes_decay_rate
        return 1 / decay_rates * np.log(90. / 100.)

    def plot_eigfn_dynamics(self, mode_indices: Optional[list[int]] = None):
        """ Create plotly (n_modes) x 2 subplots to show the eigenfunctions dynamics.

        The first column will plot the eigenfunctions in the complex plane, while the second column will plot the
        eigenfunctions real part vs time. In the second plot we will show a vertical line for the mode's transient time,
        if the decay rate is positive.

        Args:
            mode_indices:

        Returns:
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if mode_indices is None:
            mode_indices = range(self.n_modes)
        n_modes_to_show = len(mode_indices)

        # Fix plot visual parameters ====================================================================================
        width = 200
        fig_width = 3 * width  # Assuming the second column and margins approximately take up another 400 pixels.
        fig_height = width * n_modes_to_show
        vertical_spacing = width * 0.1 / fig_height

        # Compute the required data for plotting the eigenfunction dynamics ===========================================
        eigfn = self.state_eigenbasis[0]  # (time_horizon, modes)
        eigfn_pred = self.pred_state_eigenbasis[0]  # (time_horizon, modes)
        time_horizon = eigfn.shape[0]
        eigval_traj = np.asarray([self.eigvals ** t for t in range(time_horizon)])  # λ_i^t for t in [0,time_horizon)

        time = np.linspace(0, time_horizon * self.dt, time_horizon)

        fig = make_subplots(rows=self.n_modes, cols=2, column_widths=[0.33, 0.66],
                            subplot_titles=[f"Mode {i // 2}" for i in range(2 * n_modes_to_show)],
                            # vertical_spacing=vertical_spacing,
                            shared_xaxes=True, shared_yaxes=True)

        time_normalized = time / (time_horizon * self.dt)
        COLOR_SCALE = "Blugrn"
        for i, mode_idx in enumerate(mode_indices):
            is_cmplx_mode = self.is_complex_mode[mode_idx]
            eigfn_re_pred = eigfn_pred[:, mode_idx].real  # Re(λ_i^t * <u_i,z_t>)
            eigfn_im_pred = eigfn_pred[:, mode_idx].imag  # Im(λ_i^t * <u_i,z_t>)

            eigfn_re = eigfn[:, mode_idx].real
            eigfn_im = eigfn[:, mode_idx].imag

            # Plot the predicted eigenfunction dynamics in the complex plane with equal aspect ratio
            fig.add_trace(go.Scatter(x=eigfn_im_pred,
                                     y=eigfn_re_pred,
                                     mode='markers',
                                     marker=dict(color=time_normalized, colorscale=COLOR_SCALE, size=4),
                                     name=f"Mode {i}",
                                     legendgroup=f"mode{i}"),
                          row=i + 1, col=1)

            # Plot the true eigenfunction dynamics in the complex plane
            fig.add_trace(go.Scatter(x=eigfn_im,
                                     y=eigfn_re,
                                     mode='lines',
                                     line=dict(color='black', width=1),
                                     name=f"Mode {i}",
                                     legendgroup=f"mode{i}"),
                          row=i + 1, col=1)

            # Plot the real part of the eigenfunction dynamics vs time
            fig.add_trace(go.Scatter(x=time,
                                     y=eigfn_re_pred * (2 if is_cmplx_mode else 1),
                                     mode='markers',
                                     marker=dict(color=time_normalized, colorscale=COLOR_SCALE, size=4),
                                     name=f"Mode {i}",
                                     legendgroup=f"mode{i}"),
                          row=i + 1, col=2)

            # Plot the true real part of the eigenfunction dynamics vs time
            fig.add_trace(go.Scatter(x=time,
                                     y=eigfn_re * (2 if is_cmplx_mode else 1),
                                     mode='lines',
                                     line=dict(color='black', width=1),
                                     name=f"Mode {i}",
                                     legendgroup=f"mode{i}"),
                          row=i + 1, col=2)

            # Plot the area between the predicted and true real part of the eigenfunction dynamics
            fig.add_trace(go.Scatter(x=np.concatenate((time, time[::-1])),
                                     y=np.concatenate((eigfn_re_pred * (2 if is_cmplx_mode else 1),
                                                       eigfn_re[::-1] * (2 if is_cmplx_mode else 1))),
                                     fill='toself',
                                     fillcolor='rgba(0,0,0,0.1)',
                                     line=dict(color='rgba(0,0,0,0)'),
                                     name=f"Mode {i}",
                                     legendgroup=f"mode{i}"),
                          row=i + 1, col=2)


        # Set the overall layout size. Adjust the width to accommodate your 200px wide first column.
        # This width calculation is an approximation and might need tweaking based on your actual layout and margins.
        fig.update_layout(
            autosize=True,
            width=fig_width,
            height=fig_height,
            showlegend=False,  # Hide the legend
            )

        # Update y-axes of the first column only to have a fixed range for a square aspect ratio
        # This is a workaround since direct width control per subplot isn't supported.
        # You might need to adjust the range based on your data for a square appearance.
        for i in range(n_modes_to_show):
            fig.update_yaxes(row=i + 1, col=1, scaleanchor="x", scaleratio=1, )
        fig.update_xaxes(rangeslider=dict(visible=False))
        return fig


    def visual_mode_selection(self):
        from dash import Dash, dcc, html, Input, Output, callback
        import pandas as pd
        import plotly.express as px

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        modes_info = self
        # Assuming `modes_info` is an instance of `ModesInfo` class
        df = pd.DataFrame({
            "modes_idx":            range(modes_info.n_modes),
            "modes_frequencies":    modes_info.modes_frequency,
            "modes_modulus":        modes_info.modes_modulus,
            "modes_decay_rate":     modes_info.modes_decay_rate,
            "modes_transient_time": modes_info.modes_transient_time
            })

        app.layout = html.Div(
            [
                html.Div(
                    dcc.Graph(id="g1", config={"displayModeBar": False}),
                    className="four columns",
                    ),
                html.Div(
                    dcc.Graph(id="g2", config={"displayModeBar": False}),
                    className="four columns",
                    ),
                html.Div(
                    dcc.Graph(id="g3", config={"displayModeBar": False}),
                    className="four columns",
                    ),
                ],
            className="row",
            )

        def get_figure(df, y_col, selectedpoints, selectedpoints_local):
            # similar to the original get_figure function, but x_col is always "modes_idx"
            x_col = "modes_idx"
            # rest of the function remains the same

            if selectedpoints_local and selectedpoints_local["range"]:
                ranges = selectedpoints_local["range"]
                selection_bounds = {
                    "x0": ranges["x"][0],
                    "x1": ranges["x"][1],
                    "y0": ranges["y"][0],
                    "y1": ranges["y"][1],
                    }
            else:
                selection_bounds = {
                    "x0": np.min(df[x_col]),
                    "x1": np.max(df[x_col]),
                    "y0": np.min(df[y_col]),
                    "y1": np.max(df[y_col]),
                    }

            # set which points are selected with the `selectedpoints` property
            # and style those points with the `selected` and `unselected`
            # attribute. see
            # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
            # for an explanation
            fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index)

            fig.update_traces(
                selectedpoints=selectedpoints,
                customdata=df.index,
                mode="markers+text",
                marker={"color": "rgba(0, 116, 217, 0.7)", "size": 20},
                unselected={
                    "marker":   {"opacity": 0.3},
                    "textfont": {"color": "rgba(0, 0, 0, 0)"},
                    },
                )

            fig.update_layout(
                margin={"l": 20, "r": 0, "b": 15, "t": 5},
                dragmode="select",
                hovermode=False,
                newselection_mode="gradual",
                )

            fig.add_shape(
                dict(
                    {"type": "rect", "line": {"width": 1, "dash": "dot", "color": "darkgrey"}},
                    **selection_bounds
                    )
                )
            return fig


        @callback(
            Output("g1", "figure"),
            Output("g2", "figure"),
            Output("g3", "figure"),
            Input("g1", "selectedData"),
            Input("g2", "selectedData"),
            Input("g3", "selectedData"),
            )
        def callback(selection1, selection2, selection3):
            selectedpoints = df.index
            for selected_data in [selection1, selection2, selection3]:
                if selected_data and selected_data["points"]:
                    selectedpoints = np.intersect1d(
                        selectedpoints, [p["customdata"] for p in selected_data["points"]]
                        )

            return [
                get_figure(df, "modes_modulus", selectedpoints, selection1),
                get_figure(df, "modes_frequencies", selectedpoints, selection2),
                get_figure(df, "modes_decay_rate", selectedpoints, selection3),
                ]

        app.run_server(debug=False)


