from kooplearn.signal.utils import compute_mode_info
from kooplearn.abc import BaseModel
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

def spectrogram(X, modelClass:BaseModel, deltat, steps, T):
    """
    model
    deltat = 500 # number of samples for each training
    steps = 10   # step size of the moving window
    T=200         # number of time steps at each state

    """

    N = X.shape[0]
    freqs = np.zeros(((N-deltat-T)//steps, 50))
    amplitudes = np.zeros(((N-deltat-T)//steps, 50))
    phases = np.zeros(((N-deltat-T)//steps, 50))
    modulus = np.zeros(((N-deltat-T)//steps, 50))

    for i in tqdm(range((N-deltat-T)//steps)):
        X = setup_timeseries(X[i*steps:(i+1)*steps+deltat+T], T)

        model = deepcopy(modelClass)
        model.fit(X)

        infos = compute_mode_info(model,observable=lambda x:x[:,0].reshape(-1,1), deltat=0.001, xcoord=None)
        freqs[i] = infos['frequency']
        amplitudes[i] = infos['amplitude']
        modulus[i] = infos['modulus']
        phases[i] = infos['phase']
        del model
    return freqs, phases, amplitudes, modulus