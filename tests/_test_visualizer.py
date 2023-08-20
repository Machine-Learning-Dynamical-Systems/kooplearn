from kooplearn.visualizer.visualizer import Visualizer
from kooplearn._src.models.kernel import KernelReducedRank
from sklearn.gaussian_process.kernels import DotProduct
from kooplearn.data.datasets import MockData

dataset = MockData(num_features=5, rng_seed=0)
_Z = dataset.generate(None, 10)
X, Y = _Z[:-1], _Z[1:]

model = KernelReducedRank(DotProduct())
model.fit(X,Y)
eigs = model.eig()

vis = Visualizer(model)

fig = vis.plot_eigs()
fig.show()

fig = vis.plot_freqs()
fig.show()

fig = vis.plot_modes()
fig.show()