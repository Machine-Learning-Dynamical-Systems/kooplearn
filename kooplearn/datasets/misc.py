import abc
from typing import NamedTuple

import numpy as np
from tqdm import tqdm


class LinalgDecomposition(NamedTuple):
    values: np.ndarray
    x: np.ndarray
    functions: np.ndarray


# General utility classes
class DataGenerator(abc.ABC):
    @abc.abstractmethod
    def sample(self, X0: np.ndarray, T: int = 1):
        pass


class DiscreteTimeDynamics(DataGenerator):
    def sample(self, X0: np.ndarray, T: int = 1, show_progress: bool = False):
        X0 = np.asarray(X0)
        if X0.ndim == 0:
            X0 = X0[None]

        memory = np.zeros((T + 1,) + X0.shape)
        memory[0] = X0
        if show_progress:
            _iter = tqdm(range(T), desc="Generating data", unit="step", leave=False)
        else:
            _iter = range(T)

        for t in _iter:
            memory[t + 1] = self._step(memory[t])
        return memory

    @abc.abstractmethod
    def _step(self, X: np.ndarray):
        pass
