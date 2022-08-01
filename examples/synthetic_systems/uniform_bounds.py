from sys import path
path.append('../../')
from kooplearn.estimators import ReducedRank, PrincipalComponent
from kooplearn.kernels import RBF, Matern

from Logistic import LogisticMap
from Lorenz63 import Lorenz63

from tqdm import tqdm
import numpy as np
import pickle

from scipy.optimize import root_scalar

def compare_estimator_errors(
    Map,
    sample_size, 
    params,
    num_repetitions = 10, 
    num_test_points = 50000,
    sample_kwargs = {},
    bracket =[1e-7, 1e-1]
    ):
    samples = np.zeros((2, sample_size, Map.ndim, num_repetitions))
    for idx in range(num_repetitions):
        x, y = Map.sample(size=sample_size, **sample_kwargs)
        samples[0,...,idx] = (x - x.mean(axis=0)) / x.std(axis=0)
        samples[1,...,idx] = (y - y.mean(axis=0)) / y.std(axis=0)
    
    x_test, y_test = Map.sample(size=num_test_points, **sample_kwargs)
    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)
    y_test = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    def RRR_norm_f(tikhonov_reg, x, y, PCR_norm):
        return ReducedRank(**params, tikhonov_reg = tikhonov_reg).fit(x, y).norm() - PCR_norm
    
    results = {}
    for estimator in ['PCR', 'RRR']:
        results[estimator] = {}
        for quantity in ['train', 'test', 'norm']:
            results[estimator][quantity] = np.zeros(num_repetitions)

    for idx in tqdm(range(num_repetitions), desc=f'Computing errors for sample size: {sample_size}'):
        x = samples[0,...,idx]
        y = samples[1,...,idx]

        PCR_estimator = PrincipalComponent(**params).fit(x, y)
        results['PCR']['test'][idx] = PCR_estimator.risk(x_test, y_test)
        results['PCR']['train'][idx] = PCR_estimator.risk()
        results['PCR']['norm'][idx] = PCR_estimator.norm()

        root_result = root_scalar(RRR_norm_f, (x, y, results['PCR']['norm'][idx]), bracket=bracket)
        RRR_estimator = ReducedRank(**params, tikhonov_reg = root_result.root).fit(x, y)

        results['RRR']['test'][idx] = RRR_estimator.risk(x_test, y_test)
        results['RRR']['train'][idx] = RRR_estimator.risk()
        results['RRR']['norm'][idx] = RRR_estimator.norm()
        
    return results


if __name__ == '__main__':
    sample_sizes = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    num_repetitions = 100


    #Logistic Map
    kernel = RBF(length_scale=0.5)
    params = {
        'kernel': kernel,
        'backend': 'numpy',
        'rank': 3,
        'svd_solver': 'arnoldi',
        'n_oversamples': 10,
        'iterated_power': 3
    }
    Map = LogisticMap(N = 20)
    results = []
    for size_idx, sample_size in enumerate(sample_sizes):
        results.append(compare_estimator_errors(Map, sample_size, params, num_repetitions = num_repetitions, bracket = [1e-4, 1e-1]))
    pickle.dump((sample_sizes,results), open('logistic_map_results.pkl', 'wb'))
    #Lorenz 63
    kernel = Matern(length_scale=1.0, nu = 0.5)
    params = {
        'kernel': kernel,
        'backend': 'numpy',
        'rank': 10,
        'svd_solver': 'arnoldi',
        'n_oversamples': 10,
        'iterated_power': 3
    }
    Map = Lorenz63(dt = 0.1)
    results = []
    for size_idx, sample_size in enumerate(sample_sizes):
        results.append(compare_estimator_errors(Map, sample_size, params, num_repetitions = num_repetitions))
    pickle.dump((sample_sizes,results), open('lorenz63_results.pkl', 'wb'))

