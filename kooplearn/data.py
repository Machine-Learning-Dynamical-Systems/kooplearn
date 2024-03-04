import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("kooplearn")

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.nn.data import ContextsDataset, traj_to_contexts_dataset
except ImportError:
    pass


class Contexts:
    def __init__(self, contexts_data: np.ndarray):
        self.data = contexts_data
        # Shapes
        _shape = self.data.shape
        self._num_contexts = _shape[0]
        self._context_window_length = _shape[1]

        self._lookforward_window_length = lookforward_window_length
        if self.has_lookforward:
            self._lookback_window_length = (
                self._context_window_length - self._lookforward_window_length
            )
        else:
            self._lookback_window_length = self._context_window_length
        self._features_shape = _shape[2:]

    def lookback(self, lookback_len: int):
        return self.evolve_lookback(0)

    def lookforward(self, lookback_len: int):
        if self.has_lookforward:
            return self.data[:, self._lookback_window_length :]
        else:
            return None

    def evolve_lookback(self, steps: int, lookback_len: int):
        if self.has_lookforward:
            if (steps <= self._lookforward_window_length) and (steps >= 0):
                return self.data[:, slice(steps, self._lookback_window_length + steps)]
            else:
                raise ValueError(
                    f"Out of bounds. Cannot evolve the lookback window of {steps} steps, with a lookforward window of length {self._lookforward_window_length}."
                )
        else:
            if steps == 0:
                return self.data[:, slice(steps, self._lookback_window_length + steps)]
            else:
                raise ValueError(
                    "Cannot evolve the lookback window when the Contexts have no lookforward."
                )

    def _has_lookforward(self, lookback_len: int):
        if (self._lookforward_window_length is None) or (
            self._lookforward_window_length == 0
        ):
            return False
        else:
            return True

    def save(self, path: os.Pathlike):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Contexts <count={self._num_contexts},context_length={self._context_window_length},features={self._features_shape}>\n{self.data.__str__()}"


def traj_to_contexts(
    trajectory: np.ndarray,
    context_window_len: int = 2,
    time_lag: int = 1,
):
    """Convert a single trajectory to a sequence of context windows.

    Args:
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
        lookforward_window_len (int, optional): Length of the lookforward window. Defaults to 1.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Defaults to 1.

    Returns:
        np.ndarray: A sequence of context windows of shape ``(n_contexts, context_window_len, *features_shape)``.
    """
    if context_window_len < 2:
        raise ValueError(f"context_window_len must be >= 2, got {context_window_len}")

    if time_lag < 1:
        raise ValueError(f"time_lag must be >= 1, got {time_lag}")

    trajectory = np.asanyarray(trajectory)
    if trajectory.ndim == 0:
        trajectory = trajectory.reshape(1, 1)
    elif trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]

    _context_window_len = 1 + (context_window_len - 1) * time_lag
    if _context_window_len > trajectory.shape[0]:
        raise ValueError(
            f"Invalid combination of context_window_len={context_window_len} and time_lag={time_lag} for trajectory of length {trajectory.shape[0]}. Try reducing context_window_len or time_lag."
        )

    _res = np.lib.stride_tricks.sliding_window_view(
        trajectory, _context_window_len, axis=0
    )
    _res = np.moveaxis(_res, -1, 1)[:, ::time_lag, ...]

    return Contexts(_res)
