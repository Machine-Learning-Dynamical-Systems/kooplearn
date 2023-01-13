import mdshare
import pyemma
#Standard imports
import numpy as np
import pickle
from time import perf_counter
from tqdm import tqdm

import sys
sys.path.append("../../")
from kooplearn.estimators import ReducedRank
from kooplearn.kernels import RBF

pdb = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='data')
files = mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='data')

torsions_feat = pyemma.coordinates.featurizer(pdb)
torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
torsions_data = np.concatenate(pyemma.coordinates.load(files, features=torsions_feat))


def time_fn_execution(f, args):
    _start = perf_counter()
    f(*args) #time it
    _stop = perf_counter()    
    return _stop - _start

def benchmark(sample, test_sample, params, num_repetitions, solvers = ['arnoldi', 'randomized']):
    x, y = sample[:-1], sample[1:]
    x_test, y_test = test_sample[:-1], test_sample[1:]
    times = np.zeros((len(solvers), num_repetitions))
    test_errors = np.zeros((len(solvers), num_repetitions))
    training_errors = np.zeros((len(solvers), num_repetitions))
    for solver_idx, svd_solver in enumerate(solvers):
        estimator = ReducedRank(**params, svd_solver=svd_solver)
        for rep_idx in range(num_repetitions):
            times[solver_idx,rep_idx] = time_fn_execution(estimator.fit, (x,y))
            test_errors[solver_idx, rep_idx] = estimator.risk(x_test, y_test)
            training_errors[solver_idx, rep_idx] = estimator.risk()
    return times, test_errors, training_errors

if __name__ == "__main__":
    kernel = RBF(length_scale=4.0) #Sqrt(n_features)
    params = {
        'kernel': kernel,
        'backend': 'numpy',
        'rank': 5,
        'n_oversamples': 3,
        'iterated_power': 1,
        'tikhonov_reg': 1e-6 #Should be adapted to the sample size.
    }
    num_repetitions = 3
    #iter_arr = [1000, 2500, 5000, 10000, 15000, 20000]
    iter_arr = [100, 250, 500, 1000, 1500, 2000]
    solvers = ['arnoldi', 'randomized']
    test_sample = torsions_data[-1001:]

    measurement_keys = ['times', 'test_errors', 'training_errors']
    measurements = dict()
    for k in measurement_keys:
        measurements[k] = {
            'means' : np.zeros((len(solvers), len(iter_arr))),
            'stds' : np.zeros((len(solvers), len(iter_arr)))
        }

    for iter_idx, item in tqdm(enumerate(iter_arr), total=len(iter_arr)):
        times, test_errors, tranining_errors = benchmark(torsions_data[:item], test_sample, params, num_repetitions, solvers=solvers)
        for (arr, key) in zip([times, test_errors, tranining_errors], measurement_keys):
            measurements[key]['means'][:, iter_idx] = arr.mean(axis=1)
            measurements[key]['stds'][:, iter_idx] = arr.std(axis=1)

    results = {
        'iterated_array': iter_arr,
        'solvers': solvers,
        'measurements': measurements,
        'n_oversamples': params['n_oversamples'],
        'iterated_power': params['iterated_power']
    }
    pickle.dump(results, open('data/sample_randSVD_benchmarks.pkl', 'wb'))