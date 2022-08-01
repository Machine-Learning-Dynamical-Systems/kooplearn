import numpy as np
from utils import LogisticMapSimulation
import sys
sys.path.append("../../")
from kooplearn.estimators import KernelRidgeRegression, ReducedRank, PrincipalComponent, RandomizedReducedRank
from kooplearn.kernels import RBF

N = 20
kernel = RBF(length_scale = 0.5)
regularization_size = 1
tikhonov_regs = np.geomspace(1e-6, 1e-3, num=regularization_size)

parameters = {
    'num_train_samples' :   10000,
    'ranks' :               3,
    'tikhonov_regs':        tikhonov_regs,
    'estimators' :          [KernelRidgeRegression, ReducedRank],
}

# parameters = {
#     'num_train_samples' :   10000,
#     'ranks' :               3,
#     'tikhonov_regs':        tikhonov_regs,
#     'estimators' :          [PrincipalComponent],
# }

statistics = {
    'num_test_samples' : 500,
    'test_repetitions' : 1,
    'train_repetitions' : 100
}

if __name__ == "__main__":
    simulation = LogisticMapSimulation(kernel, parameters, statistics, N = N)
    train_errors, test_errors, eigvals = simulation.run_eigs(backend='keops', num = 10, save=True, save_path = "../../../data/noisy_logistic_map/")
