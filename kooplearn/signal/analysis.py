from kooplearn._src.dashboard.utils import compute_mode_info
from kooplearn._src.models.abc import BaseModel
import numpy as np
from tqdm import tqdm 
from utils import get_XY, compute_mode_info

identity = lambda x : x 

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
        X,Y = get_XY(X[i*steps:(i+1)*steps+deltat+T], T)

        model = modelClass(kernel=0.3*DotProduct()+0.7*RBF(length_scale=0.1), rank=50)
        model.fit(X,Y)

        infos = compute_mode_info(model,observable=lambda x:x[:,0].reshape(-1,1), deltat=0.001, xcoord=None)
        freqs[i] = infos['frequency']
        amplitudes[i] = infos['amplitude']
        modulus[i] = infos['modulus']
        phases[i] = infos['phase']
        del model
    return freqs, phases, amplitudes, modulus

def compute_coherence_metric(model, observable_1=identity, observable_2=identity, min_freq=None, max_freq=None, aggregation=None):
    # compute mode 1 and frequencies
    infos_1 = compute_mode_info(model, observable=observable_1)
    n_features_1 = infos_1['var_index'].unique().shape[0]

    # compute mode 2 and frequencies
    infos_2 = compute_mode_info(model, observable=observable_2)
    n_features_2 = infos_2['var_index'].unique().shape[0]

    # Selecting eigenvalues within the frequency band
    if min_freq is None:
        min_freq = infos_1['frequency'].min()
    if max_freq is None:
        max_freq = infos_1['frequency'].max()

    eigs = infos_1['eig_num'].unique()
    freqs = infos_1['frequency'].unique()
    band = (freqs <= max_freq & freqs >= min_freq)
    freqs = freqs[band]
    eigs = eigs[band]  

    # creating the coherence tensor
    coherence_tensor=np.zeros((eigs.shape[0], n_features_1, n_features_2))

    for i,eig in enumerate(eigs):
        
        # matrix for mode 1
        modes_1 = np.abs(infos_1[infos_1['eig_num']==eig]['mode'].to_numpy()) 
        modes_1 = np.repeat(modes_1, n_features_2).reshape(coherence_tensor.shape[1:]).T
        
        # matrix for mode 2
        modes_2 = np.abs(infos_2[infos_2['eig_num']==eig]['mode'].to_numpy()) 
        modes_2 = np.repeat(modes_2, n_features_1).reshape(coherence_tensor.shape[1:])
        
        # computing max and min
        max_modes = np.maximum(modes_1, modes_2)
        min_modes = np.minimum(modes_1, modes_2)

        coherence_tensor[i]= min_modes/max_modes

    if aggregation is None: return coherence_tensor
    return aggregation(coherence_tensor)

def compute_phase_delay(model, observable_1=identity, observable_2=identity, min_freq=None):
    
    # compute mode 1 and frequencies
    infos_1 = compute_mode_info(model, observable=observable_1)
    n_features_1 = infos_1['var_index'].unique().shape[0]

    # compute mode 2 and frequencies
    infos_2 = compute_mode_info(model, observable=observable_2)
    n_features_2 = infos_2['var_index'].unique().shape[0]

    # Selecting eigenvalues within the frequency band
    if min_freq is None:
        min_freq = infos_1['frequency'].min()
    if max_freq is None:
        max_freq = infos_1['frequency'].max()

    eigs = infos_1['eig_num'].unique()
    freqs = infos_1['frequency'].unique()
    band = (freqs <= max_freq & freqs >= min_freq)
    freqs = freqs[band]
    eigs = eigs[band]  

    # creating the coherence tensor
    coherence_tensor=np.zeros((eigs.shape[0], n_features_1, n_features_2))

    for i,eig in enumerate(eigs):
        
        # matrix for mode 1
        modes_1 = infos_1[infos_1['eig_num']==eig]['phase'].to_numpy()/np.pi
        modes_1 = np.repeat(modes_1, n_features_2).reshape(coherence_tensor.shape[1:]).T
        
        # matrix for mode 2
        modes_2 = infos_2[infos_2['eig_num']==eig]['phase'].to_numpy()/np.pi
        modes_2 = np.repeat(modes_2, n_features_1).reshape(coherence_tensor.shape[1:])
        
        # comparing phases (and casting in [0, 2) )
        coherence_tensor[i] = int((modes_1 - modes_2)/(2))
        coherence_tensor[i] = (modes_1 - modes_2) - coherence_tensor[i]*2

    return coherence_tensor