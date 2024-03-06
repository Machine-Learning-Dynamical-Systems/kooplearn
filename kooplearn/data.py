import os

import numpy as np
import torch
from numpy.typing import ArrayLike

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.nn.data import ContextsDataset, traj_to_contexts_dataset
except ImportError:
    pass


class TensorContextDataset(ContextWindowDataset):
    def __init__(self, data: ArrayLike):
        if data.ndim < 3:
            raise ShapeError(
                f"Invalid shape {data.shape}. The data must have be at least three dimensional [batch_size, context_len, *features]."
            )
        self.data = data
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self._context_length = self.shape[1]

    def slice(self, slice_obj):
        return self.data[:, slice_obj]


class TrajectoryContextDataset(TensorContextDataset):
    def __init__(
        self, trajectory: ArrayLike, context_length: int = 2, time_lag: int = 1
    ):
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        # It should be converted to Numpy
        trajectory = np.asanyarray(trajectory)
        if trajectory.ndim < 1:
            raise ShapeError(
                f"Invalid trajectory shape {trajectory.shape}. The trajectory should be at least 1D."
            )
        elif trajectory.ndim == 1:
            trajectory = trajectory[:, None]  # Add a dummy dimension for 1D features
        else:
            pass

        self.data, self.idx_map = ...
        self.trajectory = trajectory
        self.time_lag = time_lag
        # Shapes
        assert context_length == self.context_length

    def _build_contexts_np(self, trajectory, context_length, time_lag):
        window_shape = 1 + (context_length - 1) * time_lag
        if window_shape > trajectory.shape[0]:
            raise ValueError(
                f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of "
                f"length {trajectory.shape[0]}. Try reducing context_length or time_lag."
            )

        data = np.lib.stride_tricks.sliding_window_view(
            trajectory, window_shape, axis=0
        )

        idx_map = np.lib.stride_tricks.sliding_window_view(
            np.arange(trajectory.shape[0], dtype=np.int_), window_shape, axis=0
        )

        idx_map = np.moveaxis(idx_map, -1, 1)[:, ::time_lag, ...]
        data = np.moveaxis(data, -1, 1)[:, ::time_lag, ...]
        return data, idx_map

    def _build_contexts_torch(self, trajectory, context_length, time_lag):
        window_shape = 1 + (context_length - 1) * time_lag
        if window_shape > trajectory.shape[0]:
            raise ValueError(
                f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of length {trajectory.shape[0]}. Try reducing context_length or time_lag."
            )

        data = trajectory.unfold(0, window_shape, 1)
        idx_map = torch.arange(len(trajectory)).unfold(0, window_shape, 1)

        data = torch.movedim(data, -1, 1)[:, ::time_lag, ...]
        idx_map = torch.movedim(idx_map, -1, 1)[:, ::time_lag, ...]
        return data, idx_map


class MultiTrajectoryContextDataset(TrajectoryContextDataset):
    def __init__(self):
        raise NotImplementedError


def traj_to_contexts(
    trajectory: np.ndarray,
    context_window_len: int = 2,
    time_lag: int = 1,
):
    """Convert a single trajectory to a sequence of context windows.

    Args:
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Defaults to 1.

    Returns:
        TrajectoryContexts: A sequence of Context Windows.
    """

    return TrajectoryContextDataset(
        trajectory, context_length=context_window_len, time_lag=time_lag
    )
