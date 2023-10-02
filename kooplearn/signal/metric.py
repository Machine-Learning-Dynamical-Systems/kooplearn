from kooplearn.dashboard.utils import compute_mode_info
from kooplearn.abc import BaseModel
import numpy as np
from tqdm import tqdm 
from utils import get_XY, compute_mode_info

identity = lambda x : x 

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