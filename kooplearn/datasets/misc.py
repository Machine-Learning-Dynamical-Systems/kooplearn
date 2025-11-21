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
    def sample(self, initial_condition: np.ndarray, T: int = 1):
        pass


class DiscreteTimeDynamics(DataGenerator):
    def sample(
        self, initial_condition: np.ndarray, T: int = 1, show_progress: bool = False
    ):
        """Base Class to implement a discrete-time dataset

        Args:
            initial_condition (np.ndarray): Initial condition
            T (int, optional): Number of steps of the evolution. Defaults to 1.

        Returns:
            np.ndarray: An array of shape ``(T + 1, *initial_condition.shape)``
        """
        initial_condition = np.asarray(initial_condition)
        if initial_condition.ndim == 0:
            initial_condition = initial_condition[None]

        memory = np.zeros((T + 1,) + initial_condition.shape)
        memory[0] = initial_condition
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
