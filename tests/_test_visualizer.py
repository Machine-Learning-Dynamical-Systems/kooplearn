from kooplearn.visualizer.visualizer import Visualizer
from kooplearn.models.kernel import KernelReducedRank
from kooplearn._src.kernels import Linear
from kooplearn.data.datasets import MockData

dataset = MockData(num_features=5, rng_seed=0)
_Z = dataset.generate(None, 10)
X, Y = _Z[:-1], _Z[1:]

model = KernelReducedRank(Linear())
model.fit(X,Y)
eigs = model.eig()

vis = Visualizer(model)

fig = vis.plot_eigs()
fig.show()

fig = vis.plot_freqs()
fig.show()

fig = vis.plot_modes()
fig.show()