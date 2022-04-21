import numpy as np
import pandas as pd

#Timescales in units of timestep for given eigenvalues
def eigenvalues_timescale(eigenvalues, stride, timestep):
    eps = 1e-12
    data = []
    for ev in eigenvalues:
        decay_length = np.around((-np.log(np.abs(ev))**-1)*stride*timestep, decimals = 2)
        if np.abs(np.imag(ev)) < eps:
            #Real eigenvalue
            freq = None
            data.append((decay_length, freq))
        else:
            #Imaginary eigenvalue
            freq = np.around(2*np.pi*(np.angle(ev)**-1)*stride*timestep, decimals=2)
            data.append((decay_length, freq))
    #Data format: (decay_length, frequency)
    return data

#Print timescales in units of timestep for given eigenvalues
def print_timescale(eigenvalues, stride, timestep, units=''):
    data = eigenvalues_timescale(eigenvalues, stride, timestep) 
    col_width = max(len(str(_x[0]) + " " + units) for _x in data) + 2  # padding
    for x in data:
        if x[1] == None:
            print(f"Decay:{x[0]} {units}")
        elif x[1] <= 0:
            pass
        else:
            a = str(x[0]) + " " + units
            print("Decay: " + a.ljust(col_width) + f"Frequency: {x[1]} {units}")

def extract_windows_vectorized(array,sub_window_size):
    #Sliding window every timestep, adapted from https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    max_time = array.shape[0] - sub_window_size
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )
    res = array[sub_windows]
    return res.reshape(res.shape[0], -1)

def subsample_data(df, n_timesteps, shift=0):
    #Select 1 data point every n_timesteps
    raised_exception = False
    dataset_indexes = []
    i = 0
    while not raised_exception:
        try:
            df[shift + i*n_timesteps]
            dataset_indexes.append(shift + i*n_timesteps)
            i += 1
        except Exception as e:
            raised_exception = True
    return df[np.array(dataset_indexes)]

def get_training_dataset(df, measurements, n_timesteps = None, history = None, average = False, shift = 0):
    if average:
        features_names = [m + "_mean" for m in measurements]
        _df = np.hstack([df[m][1].mean(axis=1)[:, np.newaxis] for m in measurements])
    else:
        features_names = [elem for sublist in [df[m][0] for m in measurements] for elem in sublist]
        _df = np.hstack([df[m][1] for m in measurements])
    #Applying subsampling and then sliding window
    if n_timesteps is not None:
        _df = subsample_data(_df, n_timesteps, shift = shift)
    if history is not None:
        _df = extract_windows_vectorized(_df, history)
        features_names_old = features_names
        features_names = [features_names_old[i] + " + " + str(j) for j in range(history) for i in range(len(features_names_old))]
    #Return dataframe as numpy array and features names
    return _df, features_names

#If history, mask everything but the last history timesteps
def mask_old_timesteps(array, features_names, measurement, history):
    idxs = []
    num_cols = array.shape[1]
    num_features = int(num_cols/history)
    for i,x in enumerate(features_names):
        if measurement in x:
            if i >= num_features*int(history - 1):
                idxs.append(i)
    return array[:, idxs]

def prepare_dataset(df_path):
    df_pd = pd.read_pickle(df_path).interpolate().dropna()
    #Get stations and measurements 
    measurements = set()
    for col in df_pd.columns:
        meas, stat = col.split("_")
        measurements.add(meas)

    standardizing_parameters = dict() # (std, mean)

    for m in measurements:
        std = df_pd.filter(regex= m + "_*").std().mean()
        mean = df_pd.filter(regex= m + "_*").mean().mean()
        standardizing_parameters[m] = (std, mean)

    #Scale measurements
    df = dict()
    for m in measurements:
        df[m] = (list(df_pd.filter(regex= m + "_*").columns), df_pd.filter(regex= m + "_*").apply(lambda x: scale(x, standardizing_parameters[m][1], standardizing_parameters[m][0])).to_numpy())

    return df, measurements, standardizing_parameters

def scale(x, m, s):
        return (x - m)/s

def unscale(x, m, s):
    return x*s + m