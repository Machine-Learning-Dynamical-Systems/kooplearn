import numpy as np
import pandas

def compute_mode_info(operator, observable=lambda x:x, X=None, deltat=1, xcoord=None, ycoord=None):
    eigs = operator.eig()
    if X is None:
        X = operator.X_fit[-1].reshape(1, -1)
    modes = operator.modes(X, observables=observable)
    n_eigs = eigs.shape[0]
    if modes.ndim == 1:
        n_features=1
    else:
        n_features = modes.shape[1]
        print(n_features)

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
        Z = modes*eigs
    else:
        Z = modes * np.outer(eigs, np.ones(n_features))  # Multiplying by the eigenvalue to recover the signal
    Z = Z.flatten()  # Row-wise flattening of the modes matrix
    infos['amplitude'] = np.abs(Z) # Amplitude of the mode at every point
    infos['phase'] = np.arctan(Z.imag / Z.real)  # Phase of the mode at every point

    return infos

def get_XY(Z, T=1, order='F'):

    # concatenates data so that Xi are (A_t, A_t-1, ... A_t-T, B_t, B_t-1, ..., B_t-T)
    Zbis = []
    for i in range(len(Z)-T):
        Zi = Z[i:i+T].flatten(order) # 
        Zbis.append(Zi)
    Zbis = np.array(Zbis)
    X = Zbis[:-1]
    Y = Zbis[1:]
    return X,Y