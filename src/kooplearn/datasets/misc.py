import abc
from typing import NamedTuple

import numpy as np
import pandas as pd


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
    def __init__(self, dt: float = 1.0):
        self.dt = dt

    @abc.abstractmethod
    def _step(self, X: np.ndarray):
        pass

    def sample(self, X0: np.ndarray, T: int = 1) -> pd.DataFrame:
        X0 = np.atleast_1d(X0)
        t_eval = np.arange(0, T + self.dt, self.dt)
        memory = np.zeros((len(t_eval),) + X0.shape)
        memory[0] = X0

        for t in range(len(t_eval) - 1):
            memory[t + 1] = self._step(memory[t])

        # MultiIndex: step + time
        step_index = np.arange(len(t_eval))
        index = pd.MultiIndex.from_arrays([step_index, t_eval], names=["step", "time"])

        # columns = [f"x{i}" for i in range(memory.shape[1])] if memory.ndim > 1 else ["x"]
        traj = pd.DataFrame(memory, columns=self.df.columns, index=index)
        traj.attrs = self.df.attrs
        traj.attrs["X0"] = X0
        self.df = traj

        return self.df


class DataFrameMixin:
    """Mixin to manage an internal DataFrame for generated trajectories."""

    def _init_dataframe(self, columns):
        """Initialize (or reset) the internal DataFrame with metadata."""
        self.df = pd.DataFrame(columns=columns)
        self.df.attrs = {
            "generator": self.__class__.__name__,
            "params": self._get_params(),
        }

    def _update_dataframe(self, data, index):
        df = pd.DataFrame(data, columns=self.df.columns, index=index)
        df.attrs = self.df.attrs
        self.df = df

    def _get_params(self):
        """Collect relevant model parameters for storage in attrs."""
        params = {}
        for k, v in self.__dict__.items():
            if k in {"df", "rng"} or k.startswith("_"):
                continue
            params[k] = v
        return params
