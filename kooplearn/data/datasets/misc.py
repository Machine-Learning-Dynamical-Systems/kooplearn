import abc
from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike

class LinalgDecomposition(NamedTuple):
    values: ArrayLike
    x: ArrayLike
    functions: ArrayLike
    
#General utility classes
class DataGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, X0: ArrayLike, T:int=1):
        pass

class DiscreteTimeDynamics(DataGenerator):
    def generate(self, X0: ArrayLike, T:int=1):
        memory = np.zeros((T+1, X0.shape))
        memory[0] = X0
        for t in range(T):
            memory[t+1] = self._step(memory[t])
        return memory
    @abc.abstractmethod
    def _step(self, X: ArrayLike):
        pass

