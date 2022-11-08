from sys import path
path.append('../../')
import argparse
from kooplearn.estimators import ReducedRank
from kooplearn.kernels import RBF, Matern, Linear

from Logistic import LogisticMap
from Lorenz63 import Lorenz63
from NoisyLinearSystem import NoisyLinear

from tqdm import tqdm
import numpy as np
import scipy
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
def error_bound(x, y, params, num_repetitions, randomized_weighted_sampling = False):
    estimator = ReducedRank(**params, svd_solver='full')
    estimator.fit(x, y, _save_svals = True, _randomized_weighted_sampling = randomized_weighted_sampling)
    svals_sq = estimator.fit_sq_svals_
    evals = estimator.eig(left=False, right=False)
    risk_full = estimator.risk()
    if randomized_weighted_sampling:
        covariance_norm = None
    else:
        inv_dim = estimator.K_X_.shape[0]**-1
        covariance_norm = inv_dim*np.linalg.norm(estimator.K_Y_, ord=2)

    estimator = ReducedRank(**params, svd_solver='randomized')
    risk_delta = np.zeros(num_repetitions)
    for rep_idx in range(num_repetitions):
        estimator.fit(x, y)
        risk_delta[rep_idx] = estimator.risk() - risk_full
    return theoretical_error_estimate(svals_sq, params['rank'], params['n_oversamples'], params['iterated_power'], covariance_norm), risk_delta, svals_sq, evals
def theoretical_error_estimate(svals_sq, rank, n_oversamples, iterated_power, covariance_norm = None):
    #Variable renaming to be consistend with paper notation (Theorems 2 and 3).
    svals_sq = np.sort(svals_sq)[::-1]
    r = rank
    s = float(n_oversamples)
    p = iterated_power
    if covariance_norm is not None:
        L = covariance_norm
        svals_normed = svals_sq/svals_sq[r-1] #renormalization by sigma_{r + 1}
        svals_pre = svals_normed[:r]**-1 #Invert to use in the def of a_r and b_r
        svals_post = svals_normed[r:]
        c_r = np.sum(svals_post**(2*p))*L
        a_r = (svals_sq[r-1]**-1)*c_r*( 1 + ((s - 1)**(-1))*np.sum(svals_pre**(2*p + 1)))
        b_r = c_r*( svals_sq[0]/svals_sq[r-1] + ((s - 1)**(-1))*np.sum(svals_pre**(2*p)))
    else:
        svals_normed = svals_sq/svals_sq[r] #renormalization by sigma_{r + 1}
        svals_pre = svals_normed[:r]**-1 #Invert to use in the def of a_r and b_r
        svals_post = svals_normed[r:]
        a_r = ((s - 1)**(-1))*np.sum(svals_pre**(2*p + 1))*np.sum(svals_post**(2*p + 1))
        b_r = (svals_sq[r])*((s - 1)**(-1))*np.sum(svals_pre**(2*p))*np.sum(svals_post**(2*p + 1))
    a = (r*a_r*svals_sq[0])/(r + a_r)
    return np.minimum(a, b_r)

if __name__ == '__main__':
    seed = 0

    num_repetitions = 3
    sample_size = 1000

    params = {
        'kernel': Linear(coef0=0),
        'backend': 'numpy',
        'tikhonov_reg': 1e-8,
        'rank': 15,
        'n_oversamples': 10,
        'iterated_power': 2
    }
    """
    Generation of the matrix A for the noisy linear example
    """
    ndim = 100
    num_ones = 10
    U = scipy.stats.special_ortho_group.rvs(ndim, random_state=seed)
    eigenvalues = np.concatenate([np.ones(num_ones), np.arange(2, ndim - num_ones + 2)**-0.5])
    
    A = U.dot(np.diag(eigenvalues)).dot(U.T)
    ######################

    map = NoisyLinear(stability = 1 - 1e-4, A = A)
    target_ranks = np.arange(10, dtype=int) + 5

    x, y = map.sample(size=sample_size)

    results = []
    for randomized_weighted_sampling in [True, False]:
        theoretical_estimate = np.zeros(target_ranks.shape[0])
        empirical_estimate_mean = np.zeros(target_ranks.shape[0])
        empirical_estimate_std = np.zeros(target_ranks.shape[0])
        for rk_idx, rank in enumerate(target_ranks):
            params['rank'] = rank
            th_estimate, risk_delta, svals_sq, evals = error_bound(x, y, params, num_repetitions, randomized_weighted_sampling=randomized_weighted_sampling)
            theoretical_estimate[rk_idx] = th_estimate
            empirical_estimate_mean[rk_idx] = np.mean(risk_delta)
            empirical_estimate_std[rk_idx] = np.std(risk_delta)
        
        results.append({
            'target_ranks': target_ranks.copy(),
            'th_estimate': theoretical_estimate.copy(),
            'means': empirical_estimate_mean.copy(),
            'stds': empirical_estimate_std.copy(),
            'svals_sq': svals_sq,
            'evals': evals
        })
    pickle.dump(results, open('data/randSVD_errorbounds.pkl', 'wb'))

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")

    args = parser.parse_args()
    seed = 0 #For reproducibility

    sample_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000]
    regularizations = np.geomspace(1e-9, 1e-3, 10)

    if args.mode == 'sample':
        iter_arr = sample_sizes
    elif args.mode == 'reg':
        iter_arr = regularizations
    else:
        raise ValueError("Available modes are 'sample' and 'reg'")
    
    reg_0 = 1e-5
    sample_0 = 10000

    solvers = ['randomized', 'arnoldi']
    
    base_params = {
        'backend': 'numpy',
        'n_oversamples': 5,
        'iterated_power': 1
    }
    num_repetitions = 5

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
        },
        {
            #NoisyLinear
            'kernel': Linear(coef0=0),
            'rank': 10,
        }
    ]

    ### Generation of the matrix A for the noisy linear example
    ndim = 50
    random_basis_change = scipy.stats.special_ortho_group.rvs(ndim, random_state=seed)
    temperature = 0.2
    eigenvalues = 0.5*(1 - np.tanh(np.linspace(-2, 2, ndim)/temperature))
    A = random_basis_change.T.dot(np.diag(eigenvalues)).dot(random_basis_change)
    ######################

    maps = [
        LogisticMap(N = 20, seed=seed), 
        Lorenz63(dt = 0.1, seed=seed),
        NoisyLinear(stability = 0.999, A = A)
    ]

    file_names = ['logistic_map', 'lorenz63', 'noisylinear']

    for (params, map, file_name) in zip(map_specific_params, maps, file_names):
        if file_name == 'noisylinear':
            means = np.zeros((len(solvers), len(iter_arr)))
            stds = np.zeros((len(solvers), len(iter_arr)))

            for iter_idx, item in tqdm(enumerate(iter_arr), total=len(iter_arr), desc=file_name):
                if args.mode == 'sample':
                    sample_size = item
                    tikhonov_reg = reg_0
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
"""