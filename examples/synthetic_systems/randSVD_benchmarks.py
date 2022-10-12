from sys import path
path.append('../../')
from kooplearn.estimators import ReducedRank, PrincipalComponent
from kooplearn.kernels import RBF, Matern

from Logistic import LogisticMap
from Lorenz63 import Lorenz63

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import pickle


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 9
})

def time_fn_execution(f, args, num_repetitions):
    times = np.zeros(num_repetitions)
    for rep_idx in range(num_repetitions):
        _start = perf_counter()
        f(*args) #time it
        _stop = perf_counter()
        times[rep_idx] = _stop - _start
    return times

def benchmark(map, sample_size, params, num_repetitions, sample_kwargs = {}, solvers = ['full', 'randomized']):
    x, y = map.sample(size=sample_size, **sample_kwargs)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    times = np.zeros((len(solvers), num_repetitions))
    for solver_idx, svd_solver in enumerate(solvers):
        estimator = ReducedRank(**params, svd_solver=svd_solver)
        times[solver_idx,:] = time_fn_execution(estimator.fit, (x,y), num_repetitions)
    return times

if __name__ == '__main__':
    seed = 0 #For reproducibility
    sample_sizes = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    #sample_sizes = [100, 200, 300, 400, 500, 600, 700]
    
    solvers = ['full', 'randomized', 'arnoldi']

    num_repetitions = 5

    map_specific_params = [
        {
            #Logistic Map
            'kernel': RBF(length_scale=0.5),
            'backend': 'numpy',
            'rank': 3,
            'tikhonov_reg': 1e-6,
            'n_oversamples': 10,
            'iterated_power': 3
        },
        {
            #Lorenz63
            'kernel': Matern(length_scale=1.0, nu = 0.5),
            'backend': 'numpy',
            'rank': 10,
            'tikhonov_reg': 1e-6,
            'n_oversamples': 10,
            'iterated_power': 3
        }
    ]
    maps = [LogisticMap(N = 20, seed=seed), Lorenz63(dt = 0.1, seed=seed)]
    file_names = ['logistic_map', 'lorenz63']

    for (params, map, file_name) in zip(map_specific_params, maps, file_names):
        means = np.zeros((len(solvers), len(sample_sizes)))
        stds = np.zeros((len(solvers), len(sample_sizes)))

        for sample_idx, sample_size in tqdm(enumerate(sample_sizes), total=len(sample_sizes), desc=file_name):
            times = benchmark(map, sample_size, params, num_repetitions, solvers=solvers)
            means[:, sample_idx] = times.mean(axis=1)
            stds[:, sample_idx] = times.std(axis=1)

        results = {
            'sample_sizes': sample_sizes,
            'solvers': solvers,
            'means': means,
            'stds': stds
        }
        pickle.dump(results, open('data/' + file_name + '_randSVD_benchmarks.pkl', 'wb'))