from DirectEstimators import DirectRegressor
from KernelEstimators import KernelReducedRank, KernelPrincipalComponent
from .._legacy_code.kernels import Linear
from scipy.stats import ortho_group
import numpy as np

# Test 1. comparing linear kernel and direct estimation

estimator1 = DirectRegressor()
estimator2 = KernelPrincipalComponent(Linear())
estimator3 = KernelReducedRank(Linear())

d=5
N=1000
# generating synthetic deterministic trajectories
A = ortho_group.rvs(d)
X0 = np.random.uniform(-1, 1, d)
X = np.zeros((N, d))
X[0] = X0
for i in range(N-1):
    X[i+1] = A @ X[i]

Y = X[1:]
X = X[:-1]

estimator1.fit(X,Y)
estimator2.fit(X,Y)
estimator3.fit(X,Y)

# comparing modes