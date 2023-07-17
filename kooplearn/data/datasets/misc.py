import abc
from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm


class LinalgDecomposition(NamedTuple):
    values: ArrayLike
    x: ArrayLike
    functions: ArrayLike


# General utility classes
class DataGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, X0: ArrayLike, T: int = 1):
        pass


class DiscreteTimeDynamics(DataGenerator):
    def generate(self, X0: ArrayLike, T: int = 1, show_progress: bool = False):
        memory = np.zeros((T + 1,) + X0.shape)
        memory[0] = X0
        if show_progress:
            _iter = tqdm(range(T), desc='Generating data', unit='step', leave=False)
        else:
            _iter = range(T)

        for t in _iter:
            memory[t + 1] = self._step(memory[t])
        return memory

    @abc.abstractmethod
    def _step(self, X: ArrayLike):
        pass
