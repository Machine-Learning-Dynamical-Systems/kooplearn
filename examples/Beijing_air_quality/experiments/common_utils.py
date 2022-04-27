from curses import window
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

def extract_windows_vectorized(array,window_size=1):
    #Sliding window every timestep, adapted from https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    max_time = array.shape[0] - window_size + 1
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(max_time), 0).T
    )
    res = array[sub_windows]
    return res.reshape(res.shape[0], -1)

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

def split_by_measurement_and_normalize(df_path):
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
        df[m] = df_pd.filter(regex= m + "_*").apply(lambda x: scale(x, standardizing_parameters[m][1], standardizing_parameters[m][0]))

    return df, standardizing_parameters

def prepare_training_dataset(df, n_timesteps = 1, history = 1, average = False, shift = 0):
    measurements = list(df.keys())
    if average:
        dfs = [df[m].mean(axis='columns').rename(m) for m in measurements]
        _df = pd.concat(dfs, axis=1)
    else:
        _df = pd.concat([d for d in df.values()], axis=1)
    #Applying subsampling and then sliding window
    _df = _df.iloc[shift::n_timesteps, :]
    _df_np = extract_windows_vectorized(_df.to_numpy(), history)
    cols = list(_df.columns)
    features_names = []
    for i in range(history):
        for c in cols:
            if i == history - 1:
                features_names.append(c)
            else:
                features_names.append(c + " " + str((i - history + 1)*n_timesteps))

    #X, Y, features names
    return _df_np[:-1], _df_np[1:], features_names

def scale(x, m, s):
        return (x - m)/s

def unscale(x, m, s):
    return x*s + m