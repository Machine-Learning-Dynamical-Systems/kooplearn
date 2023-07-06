from kooplearn.visualizer.utils import create_plot_eigs, create_frequency_plot
from kooplearn.models.kernel import KernelReducedRank
from kooplearn._src.kernels import Linear
from kooplearn.data.datasets import MockData

dataset = MockData(num_features=5, rng_seed=0)
_Z = dataset.generate(None, 10)
X, Y = _Z[:-1], _Z[1:]

model = KernelReducedRank(Linear())
model.fit(X,Y)
eigs = model.eig()

fig = create_plot_eigs(eigs)
fig.show()

fig = create_frequency_plot(eigs)
fig.show()