from kooplearn.signal.utils import compute_mode_info
from kooplearn.abc import BaseModel
from kooplearn.data import traj_to_contexts
import numpy as np
from tqdm import tqdm 
from copy import deepcopy

def setup_timeseries(Z, lookback=1, forward=1):
    # in the lookback paradigm
    X = np.zeros((Z.shape[0] - lookback - forward, lookback+forward, Z.shape[1]))
    for i in tqdm(range(X.shape[0])):
        Zi = Z[i:i+lookback+forward]
        X[i] = Zi
    return X

def spectrogram(X, modelClass:BaseModel, deltat, steps, T, observable=lambda x:x):
    """
    X one dimensional time series
    model
    deltat = 500 # number of samples for each training
    steps = 10   # step size of the moving window
    T=200         # number of time steps at each state

    """

    N = X.shape[0]
    r = modelClass.rank
    features = observable(X).shape[-1]
    freqs = np.zeros(((N-deltat-T)//steps, r*features))
    amplitudes = np.zeros(((N-deltat-T)//steps, r*features))
    phases = np.zeros(((N-deltat-T)//steps, r*features))
    modulus = np.zeros(((N-deltat-T)//steps, r*features))

    if len(X.shape) < 3:
        X = traj_to_contexts(X, T+1)

    for i in tqdm(range((N-deltat-T)//steps)):

        model = deepcopy(modelClass)
        model.fit(X[i*steps:(i+1)*steps+deltat], verbose=False)

        infos = compute_mode_info(model,observable, deltat=0.001, xcoord=None)
        freqs[i] = infos['frequency']
        amplitudes[i] = infos['amplitude']
        modulus[i] = infos['modulus']
        phases[i] = infos['phase']
        del model
    return freqs, phases, amplitudes, modulus