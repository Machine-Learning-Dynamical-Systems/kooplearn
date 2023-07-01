from ..kooplearn.koopman_estimators.DirectEstimators import DirectReducedRank
from ..kooplearn.koopman_estimators.KernelEstimators import KernelReducedRank, KernelPrincipalComponent
from ..kooplearn._src.kernels import Linear
from scipy.stats import ortho_group
import numpy as np

# Test 1. comparing linear kernel and direct estimation``

estimator1 = DirectReducedRank(tikhonov_reg=0.01)
estimator2 = KernelPrincipalComponent(Linear(), tikhonov_reg=0.01)
estimator3 = KernelReducedRank(Linear(), tikhonov_reg=0.01)

d=5
N=100
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

# comparing predictions
Yp_1 = estimator1.predict(X[0:2])
Yp_2 = estimator2.predict(X[0:2])
Yp_3 = estimator3.predict(X[0:2])

print(Yp_1-Y[0:2])
print(Yp_2-Y[0:2])
print(Yp_3-Y[0:2])

# comparing T=2 predictions
Yp_1 = estimator1.predict(X[0:1], t=2)
Yp_2 = estimator2.predict(X[0:1], t=2)
Yp_3 = estimator3.predict(X[0:1], t=2)

print(Yp_1-Y[1:2])
print(Yp_2-Y[1:2])
print(Yp_3-Y[1:2])