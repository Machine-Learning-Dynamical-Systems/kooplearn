import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from utils import LogisticMap, CosineKernel, LogisticMapSimulation
import sys
sys.path.append("../../")
from kooplearn.estimators import KernelRidgeRegression, ReducedRankRegression, PrincipalComponentRegression, RandomizedReducedRankRegression
from kooplearn.kernels import RBF

N = 20
kernel = CosineKernel(N)
kernel = RBF(length_scale = 0.5)
regularization_size = 30
tikhonov_regs = np.geomspace(1e-7, 1e-2, num=regularization_size)

parameters = {
    'num_train_samples' :   10000,
    'ranks' :               3,
    'tikhonov_regs':        tikhonov_regs,
    'estimators' :          [KernelRidgeRegression],
}

statistics = {
    'num_test_samples' : 500,
    'test_repetitions' : 1,
    'train_repetitions' : 100
}

if __name__ == "__main__":
    simulation = LogisticMapSimulation(kernel, parameters, statistics, N = N)
    train_errors, test_errors, eigvals = simulation.run_eigs(backend='keops', num = 10)
    np.save('_tmp_eigenvalues.npy', eigvals)
    np.save('_tmp_train_errors.npy', train_errors)
    np.save('_tmp_test_errors.npy', test_errors)

