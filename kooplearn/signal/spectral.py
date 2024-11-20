import numpy as np
import pandas
from kooplearn.abc import BaseModel
from kooplearn.data import traj_to_contexts
from copy import deepcopy
from tqdm.auto import tqdm

def compute_mode_info(operator:BaseModel, use_observables=True, X=None, deltat=1, xcoord=None, ycoord=None):
    # computes mode information from a kooplearn.basemodel

    if X is None:
        X = operator.data_fit

    # compute modes
    modes, eigs = operator.modes(X, predict_observables=use_observables)

    # modes have dimensions : (number of modes, time steps, context length, feature size)

    # if there are observables, extract them from the resulting dictionary
    # Only the last context from the last mode is used for our analysis since we study f(X_t)
    if use_observables:
        modes = modes['obs_of_interest'][:,-1, -1]

    # compute eigenvalues
    n_eigs = eigs.shape[0]
    if modes.ndim == 1:
        n_features=1
    else:
        n_features = modes.shape[-1]

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
    # angles = np.log(eigs)
    # freqs = angles.imag / (2 * np.pi * deltat)
    freqs = np.angle(eigs)/ (2 * np.pi * deltat) #angles.imag / (2 * np.pi * deltat)
    infos['frequency'] = np.repeat(freqs, n_features)  # Frequency of the mode

    # mode specific information. This information is unique per mode and per variable
    infos['mode'] = modes.flatten()  # Actual value of the mode
    if n_features == 1:
        Z = eigs*modes.flatten()
    else:
        Z = modes * np.outer(eigs, np.ones(n_features))  # Multiplying by the eigenvalue to recover the signal
    Z = Z.flatten()  # Row-wise flattening of the modes matrix
    infos['amplitude'] = np.abs(Z) # Amplitude of the mode at every point
    # infos['phase'] = np.arctan(Z.imag / Z.real)  # Phase of the mode at every point
    infos['phase'] = np.angle(Z)  # Phase of the mode at every point

    return infos

def spectrogram(trajectory, modelClass:BaseModel, window_size, steps, context_length, observable=lambda x:x, deltat=1.):
    """
    X one dimensional time series
    model
    window_size = 500 # number of samples for each training
    steps = 10   # step size of the moving window
    T=200         # number of time steps at each state
    observable : a function that applies on each time step of the trajectory (if X is of shape (N, d), a function from R^d to R^m, m being any integer)
    """
    ctxs = traj_to_contexts(trajectory, context_window_len=context_length)
    # applying the observable function to the trajectory:
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape((-1, 1))
    fX = np.apply_along_axis(observable, 1, trajectory)
    obs_ctxs = traj_to_contexts(fX, context_window_len=context_length-1)

    N = ctxs.shape[0]
    r = modelClass.rank

    # counting features of the observable:
    altered_traj = np.apply_along_axis(observable, 0, trajectory)
    if altered_traj.ndim == 1:
        n_features = 1
    else:
        n_features = altered_traj.shape[-1]
        
    freqs = np.zeros(((N-window_size)//steps, r*n_features))
    amplitudes = np.zeros(((N-window_size)//steps, r*n_features))
    phases = np.zeros(((N-window_size)//steps, r*n_features))
    modulus = np.zeros(((N-window_size)//steps, r*n_features))
    time = np.zeros(((N-window_size)//steps, r*n_features))

    for i in tqdm(range((N-window_size)//steps)):
        partial_dataset = ctxs[i*steps:i*steps+window_size]
        # Manually adding observables (TODO: in future versions, addind observables will be streamlined)
        partial_dataset.observables = {
            'obs_of_interest': obs_ctxs.data[i*steps:i*steps+window_size] # must have same size than training dataset, but only the last sample will be used
        }

        model = deepcopy(modelClass)
        model.fit(partial_dataset)

        infos = compute_mode_info(model, use_observables=True, deltat=deltat, xcoord=None)
        freqs[i] = infos['frequency']
        amplitudes[i] = infos['amplitude']
        modulus[i] = infos['modulus']
        phases[i] = infos['phase']
        time[i] = (i*steps+window_size)*deltat
        del model

    return freqs, phases, amplitudes, modulus, time

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

def filtered_signal(operator:BaseModel, X=None, observable=lambda x:x, modulus = (0.9,1.), angle = (0.,np.pi), nonoscilatory = False):
    eigs = operator.eig()
    if X is None:
        X = operator.data_fit.data[0, :operator.lookback_len].reshape(1, operator.lookback_len, -1)

    modes = operator.modes(X)
    n_eigs = eigs.shape[0]
    if modes.ndim == 1:
        n_features=1
    else:
        n_features = modes.shape[1]
    eigs[eigs>1.]=1.
    id = np.where(eigs.abs()>=modulus[0] & eigs.abs()<=modulus[1] & 
                  eigs.angle()>=angle[0] & eigs.angle()<=angle[1] & (eigs.angle()!=0. if not(nonoscilatory) else True))
    
    return lambda t : (eigs[id]**t)*modes.flatten() 

def compute_coherence_metric(model, observable_1=lambda x : x, observable_2=lambda x : x, min_freq=None, max_freq=None, min_modulus=None, max_modulus=None, aggregation=None):
    # compute mode 1 and frequencies
    model.data_fit.observables = {
            'obs_of_interest': observable_1(model.data_fit.data) # must have same size than training dataset, but only the last sample will be used
        }
    infos_1 = compute_mode_info(model, use_observables=True)
    n_features_1 = infos_1['var_index'].unique().shape[0]

    # compute mode 2 and frequencies
    model.data_fit.observables = {
            'obs_of_interest': observable_2(model.data_fit.data) # must have same size than training dataset, but only the last sample will be used
        }
    infos_2 = compute_mode_info(model, use_observables=True)
    n_features_2 = infos_2['var_index'].unique().shape[0]

    # Selecting eigenvalues within the frequency band
    if min_freq is None:
        min_freq = infos_1['frequency'].min()
    if max_freq is None:
        max_freq = infos_1['frequency'].max()

    eigs_freqs = infos_1[['eig_num', 'frequency']].drop_duplicates()

    eigs = eigs_freqs['eig_num']
    freqs = eigs_freqs['frequency']
    band = (freqs <= max_freq) & (freqs >= min_freq)
    freqs = freqs[band]
    eigs = eigs[band]

    # # Selecting eigenvalues within the modulus band
    # if min_modulus is None:
    #     min_modulus = infos_1['modulus'].min()
    # if max_modulus is None:
    #     max_modulus = infos_1['modulus'].max()

    # eigs_modulus = infos_1[['eig_num', 'modulus']].drop_duplicates()

    # eigs = eigs_modulus['eig_num']
    # modulus = eigs_modulus['modulus']
    # band = (modulus <= max_modulus) & (modulus >= min_modulus)
    # modulus = modulus[band]
    # eigs = eigs[band]  

    # TODO: normalise series?

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

def compute_phase_delay(model, observable_1=lambda x : x, observable_2=lambda x : x, min_freq=None, max_freq=None, min_modulus=None, max_modulus=None, aggregation=None):
    # compute mode 1 and frequencies
    model.data_fit.observables = {
            'obs_of_interest': observable_1(model.data_fit.data) # must have same size than training dataset, but only the last sample will be used
        }
    infos_1 = compute_mode_info(model, use_observables=True)
    n_features_1 = infos_1['var_index'].unique().shape[0]

    # compute mode 2 and frequencies
    model.data_fit.observables = {
            'obs_of_interest': observable_2(model.data_fit.data) # must have same size than training dataset, but only the last sample will be used
        }
    infos_2 = compute_mode_info(model, use_observables=True)
    n_features_2 = infos_2['var_index'].unique().shape[0]

    # Selecting eigenvalues within the frequency band
    if min_freq is None:
        min_freq = infos_1['frequency'].min()
    if max_freq is None:
        max_freq = infos_1['frequency'].max()

    eigs_freqs = infos_1[['eig_num', 'frequency']].drop_duplicates()

    eigs = eigs_freqs['eig_num']
    freqs = eigs_freqs['frequency']
    band = (freqs <= max_freq) & (freqs >= min_freq)
    freqs = freqs[band]
    eigs = eigs[band]  

    # # Selecting eigenvalues within the modulus band
    # if min_modulus is None:
    #     min_modulus = infos_1['modulus'].min()
    # if max_modulus is None:
    #     max_modulus = infos_1['modulus'].max()

    # eigs_modulus = infos_1[['eig_num', 'modulus']].drop_duplicates()

    # eigs = eigs_modulus['eig_num']
    # modulus = eigs_modulus['modulus']
    # band = (modulus <= max_modulus) & (modulus >= min_modulus)
    # modulus = modulus[band]
    # eigs = eigs[band]  

    # creating the coherence tensor
    coherence_tensor=np.zeros((eigs.shape[0], n_features_1, n_features_2))
    # print(coherence_tensor.shape)
    for i,eig in enumerate(eigs):
        
        # matrix for mode 1
        modes_1 = infos_1[infos_1['eig_num']==eig]['phase'].to_numpy()/(np.pi)
        modes_1 = np.repeat(modes_1, n_features_2).reshape(coherence_tensor.shape[1:]).T
        
        # matrix for mode 2
        modes_2 = infos_2[infos_2['eig_num']==eig]['phase'].to_numpy()/(np.pi)
        modes_2 = np.repeat(modes_2, n_features_1).reshape(coherence_tensor.shape[1:])
        
        # comparing phases (and casting in [0, 2) )
        #coherence_tensor[i] = int(modes_1 - modes_2)
        coherence_tensor[i][np.where(modes_2<=0)] = (modes_2 - modes_1)/2
        coherence_tensor[i][np.where(modes_2>0)] = 1. - (modes_2 - modes_1)/2
        #coherence_tensor[i] = (modes_1 - modes_2)

    return coherence_tensor