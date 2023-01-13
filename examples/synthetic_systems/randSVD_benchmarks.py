from sys import path
import copy
path.append('../../')
import argparse
from kooplearn.estimators import ReducedRank
from kooplearn.kernels import Linear

from NoisyLinearSystem import NoisyLinear

import numpy as np
import scipy
import pickle

def error_bound(x, y, params, num_repetitions):
    estimator = ReducedRank(**params, svd_solver='full')
    estimator.fit(x, y)
    svals_sq = estimator.RRR_sq_svals_
    evals = estimator.eig(left=False, right=False)
    
    if params['optimal_sketching']:
        covariance_norm = None
    else:
        inv_dim = estimator.K_X_.shape[0]**-1
        covariance_norm = inv_dim*np.linalg.norm(estimator.K_Y_, ord=2)
    estimator = ReducedRank(**params, svd_solver='randomized')
    risk_delta = np.zeros(num_repetitions)
    for rep_idx in range(num_repetitions):
        estimator.fit(x, y)
        risk_delta[rep_idx] = np.sum(svals_sq[:params['rank']]) - np.sum(estimator.RRR_sq_svals_[:params['rank']])
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

def benchmark_loop(sample, parameters, var_parameter):
    params = copy.deepcopy(parameters)
    x, y = sample
    results = []
    for optimal_sketching in [True, False]:
        params['optimal_sketching'] = optimal_sketching
        num_parameters = var_parameter['values'].shape[0]
        theoretical_estimate = np.zeros(num_parameters)
        empirical_estimate_mean = np.zeros(num_parameters)
        empirical_estimate_std = np.zeros(num_parameters)
        for idx, val in enumerate(var_parameter['values']):
            params[var_parameter['name']] = val
            th_estimate, risk_delta, svals_sq, evals = error_bound(x, y, params, num_repetitions)
            theoretical_estimate[idx] = th_estimate
            empirical_estimate_mean[idx] = np.mean(risk_delta)
            empirical_estimate_std[idx] = np.std(risk_delta)
        
        results.append({
            'th_estimate': theoretical_estimate.copy(),
            'means': empirical_estimate_mean.copy(),
            'stds': empirical_estimate_std.copy(),
            'svals_sq': svals_sq,
            'evals': evals
        })
    return results

if __name__ == '__main__':
    #Generation of the matrix A for the noisy linear example
    seed = 0
    ndim = 20
    num_ones = 10
    U = scipy.stats.special_ortho_group.rvs(ndim, random_state=seed)
    eigenvalues = np.concatenate([np.ones(num_ones), np.arange(2, ndim - num_ones + 2)**-0.05])
    A = U.dot(np.diag(eigenvalues)).dot(U.T)
    #Training data sampling
    map = NoisyLinear(stability = 0.98, A = A)
    num_repetitions = 50
    sample_size = 100
    sample = map.sample(size=sample_size)
    #Model instantiation
    parameters = {
        'kernel': Linear(coef0=0),
        'backend': 'numpy',
        'tikhonov_reg': 1e-6,
        'rank': 5,
        'n_oversamples': 5,
        'iterated_power': 1
    }

    looping_parameters = [
        {
            'name':'rank',
            'values':np.arange(10, dtype=int) + 5,
            'human_name':'Rank'
        },
        {
            'name':'n_oversamples',
            'values':np.arange(10, dtype=int) + 2,
            'human_name':r'Oversamples $s$'
        }
    ]
    results = copy.deepcopy(looping_parameters)
    for idx, var_parameter in enumerate(looping_parameters):
        print(f"Currently processing:{var_parameter['name']}")
        results[idx]['results'] = benchmark_loop(sample, parameters, var_parameter)

    pickle.dump(results, open('data/randSVD_errorbounds.pkl', 'wb'))