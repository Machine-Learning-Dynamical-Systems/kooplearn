from kooplearn.dashboard.utils import create_frequency_plot, create_plot_eigs, create_2d_plot_modes, \
    create_plot_modes, create_plot_pred, create_combined_plot_modes, create_combined_2d_plot_modes
from kooplearn.abc import BaseModel
import numpy as np
import pandas

def compute_mode_info(operator:BaseModel, observable=lambda x:x, X=None, deltat=1, xcoord=None, ycoord=None):
    # computes mode information based on the last context window?

    eigs = operator.eig()
    if X is None:
        X = operator.data_fit[-1, :operator.lookback_len].reshape(1, operator.lookback_len, -1)

    modes = operator.modes(X, observables=observable)[:,0]
    n_eigs = eigs.shape[0]
    if modes.ndim == 1:
        n_features=1
    else:
        n_features = modes.shape[1]

    # initialiasing the dataframe containing the information for every cell of every mode
    infos = pandas.DataFrame()
    infos['eig_num'] = np.repeat(np.arange(n_eigs), n_features)  # Identifying the mode number
    infos['var_index'] = np.tile(np.arange(n_features), n_eigs)  # Unique identifier for each variable
    if xcoord is None:  # If no specific coordinate is given, each dimensionis
        infos['x'] = np.tile(np.arange(n_features),
                                      n_eigs)  # identified with its index in the observable array
    else:  # Otherwise coordinates are saved for accurate plotting
        infos['x'] = xcoord
    if ycoord is not None:  # If y coordinates make sense, save them as well
        infos['y'] = ycoord

    # eigenvalue related information. This information is shared by every variable, hence the use of repeat
    infos['eigval real'] = np.repeat(eigs.real, n_features)  # Real part
    infos['eigval imag'] = np.repeat(eigs.imag, n_features)  # Imaginary part

    infos['modulus'] = np.repeat(np.abs(eigs) ** (1 / deltat), n_features)  # Modulus of the mode
    angles = np.log(eigs)
    freqs = angles.imag / (2 * np.pi * deltat)
    infos['frequency'] = np.repeat(freqs, n_features)  # Frequency of the mode

    # mode specific information. This information is unique per mode and per variable
    infos['mode'] = modes.flatten()  # Actual value of the mode
    if n_features == 1:
        Z = eigs*modes.flatten()
    else:
        Z = modes * np.outer(eigs, np.ones(n_features))  # Multiplying by the eigenvalue to recover the signal
    Z = Z.flatten()  # Row-wise flattening of the modes matrix
    infos['amplitude'] = np.abs(Z) # Amplitude of the mode at every point
    infos['phase'] = np.arctan(Z.imag / Z.real)  # Phase of the mode at every point

    return infos

class Visualizer:
    def __init__(self, koopman: BaseModel, deltat: float = 1):
        self.operator = koopman
        self.infos = None
        self.eigs = None
        self.compute_mode_info(deltat=deltat)

    def compute_mode_info(self, deltat: float, observable = lambda x : x, X0=None, xcoord=None, ycoord=None):
        if X0 is None:
            X0 = self.operator.data_fit[-1, :-1]
            X0 = X0.reshape((1, X0.shape[0], X0.shape[1]))
        self.infos = compute_mode_info(self.operator, observable, X0, deltat, xcoord, ycoord)

    def save(self, filepath: str):
        # self.operator.save(filepath)
        pass

    def load(self, filepath: str):
        # self.operator.load(filepath)
        pass

    def plot_freqs(self, min_freq=None, max_freq=None):
        fig = create_frequency_plot(self.infos, min_freq, max_freq)
        return fig

    def plot_eigs(self, min_freq=None, max_freq=None):
        if not hasattr(self, 'eigs'):
            self.eigs = self.operator.eig()
        fig = create_plot_eigs(self.infos, min_freq, max_freq)
        return fig

    def plot_modes(self, index=None, min_freq=None, max_freq=None):
        # plot moduluses and arguments
        if 'y' in self.infos.columns:
            fig = create_2d_plot_modes(self.infos, index, min_freq, max_freq)
        else:
            fig = create_plot_modes(self.infos, index, min_freq, max_freq)
        return fig

    def plot_combined_modes(self, T, min_freq=None, max_freq=None):
        # real part
        if 'y' in self.infos.columns:
            fig = create_combined_2d_plot_modes(self.infos, T, min_freq, max_freq)
        else:
            fig = create_combined_plot_modes(self.infos, T, min_freq, max_freq)
        return fig

    def plot_pred(self, X, T, min_freq=None, max_freq=None):
        preds = self.operator.predict(X, T)
        fig = create_plot_pred(self.infos, preds)
        return fig