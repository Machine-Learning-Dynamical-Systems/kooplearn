import logging
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
    def __init__(
        self, contexts_data: np.ndarray, lookback_window_length: Optional[int] = None
    ):
        self.data = contexts_data
        # Shapes
        _shape = self.data.shape
        self._num_contexts = _shape[0]
        self._context_window_length = _shape[1]
        self._lookback_window_length = lookback_window_length
        self._features_shape = _shape[1:]

    @property
    def lookback(self):
        return self.data[:, : self._lookback_window_length]

    @property
    def lookforward(self):
        if (self._lookback_window_length is None) or (
            self._lookback_window_length == self._context_window_length
        ):
            return None
        else:
            return self.data[:, self._lookback_window_length :]

    def __repr__(self):
        return f"{self._num_contexts} Contexts of length {self._context_window_length} and with {self._features_shape} features."


def traj_to_contexts(
    trajectory: np.ndarray, context_window_len: int = 2, time_lag: int = 1
):
    """Convert a single trajectory to a sequence of context windows.

    Args:
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
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
    return _res
