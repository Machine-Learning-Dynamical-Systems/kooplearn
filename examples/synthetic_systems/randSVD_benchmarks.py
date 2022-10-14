from sys import path
path.append('../../')
import argparse
from kooplearn.estimators import ReducedRank
from kooplearn.kernels import RBF, Matern

from Logistic import LogisticMap
from Lorenz63 import Lorenz63

from tqdm import tqdm
import numpy as np
from time import perf_counter
import pickle


def time_fn_execution(f, args, num_repetitions):
    times = np.zeros(num_repetitions)
    for rep_idx in range(num_repetitions):
        _start = perf_counter()
        f(*args) #time it
        _stop = perf_counter()
        times[rep_idx] = _stop - _start
    return times

def benchmark(map, sample_size, tikhonov_reg, params, num_repetitions, sample_kwargs = {}, solvers = ['full', 'randomized']):
    x, y = map.sample(size=sample_size, **sample_kwargs)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    times = np.zeros((len(solvers), num_repetitions))
    for solver_idx, svd_solver in enumerate(solvers):
        estimator = ReducedRank(**params, tikhonov_reg=tikhonov_reg, svd_solver=svd_solver)
        times[solver_idx,:] = time_fn_execution(estimator.fit, (x,y), num_repetitions)
    return times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")

    args = parser.parse_args()
    seed = 0 #For reproducibility

    sample_sizes = [100, 500, 1000, 2500, 5000, 10000]
    regularizations = np.geomspace(1e-9, 1e-3, 10)

    if args.mode == 'sample':
        iter_arr = sample_sizes
    elif args.mode == 'reg':
        iter_arr = regularizations
    else:
        raise ValueError("Available modes are 'sample' and 'reg'")
    
    reg_0 = 1e-4
    sample_0 = 10000

    solvers = ['randomized', 'arnoldi']
    
    base_params = {
        'backend': 'numpy',
        'n_oversamples': 5,
        'iterated_power': 3
    }
    num_repetitions = 2

    map_specific_params = [
        {
            #Logistic Map
            'kernel': RBF(length_scale=0.5),
            'rank': 3,           
        },
        {
            #Lorenz63
            'kernel': Matern(length_scale=1.0, nu = 0.5),
            'rank': 10,
        }
    ]
    maps = [LogisticMap(N = 20, seed=seed), Lorenz63(dt = 0.1, seed=seed)]
    file_names = ['logistic_map', 'lorenz63']

    for (params, map, file_name) in zip(map_specific_params, maps, file_names):
        means = np.zeros((len(solvers), len(iter_arr)))
        stds = np.zeros((len(solvers), len(iter_arr)))


        for iter_idx, item in tqdm(enumerate(iter_arr), total=len(iter_arr), desc=file_name):
            if args.mode == 'sample':
                sample_size = item
                tikhonov_reg = reg_0*(sample_size**-0.5)
            elif args.mode == 'reg':
                sample_size = sample_0
                tikhonov_reg = item
            
            
            times = benchmark(map, sample_size, tikhonov_reg, base_params | params, num_repetitions, solvers=solvers)
            means[:, iter_idx] = times.mean(axis=1)
            stds[:, iter_idx] = times.std(axis=1)

        results = {
            'iterated_array': iter_arr,
            'solvers': solvers,
            'means': means,
            'stds': stds,
            'n_oversamples': base_params['n_oversamples'],
            'iterated_power': base_params['iterated_power']
        }
        pickle.dump(results, open('data/' + file_name + '_randSVD_benchmarks.pkl', 'wb'))