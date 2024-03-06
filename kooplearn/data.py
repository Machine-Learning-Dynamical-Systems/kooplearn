import os

import numpy as np

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import ShapeError

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.nn.data import ContextsDataset, traj_to_contexts_dataset
except ImportError:
    pass


class ContextWindow:  # A single context window
    def __init__(self, data: np.ndarray):
        if data.ndim < 1:
            raise ShapeError(
                f"Invalid data shape {data.shape}. The context window should be at least 1D."
            )
        elif data.ndim == 1:
            data = data[:, None]  # Add a dummy dimension for 1D features
        else:
            pass

        self.data = data
        # Shapes
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self._window_length = self.shape[0]
        self._features_shape = self.shape[1:]

    @property
    def context_length(self):
        return self._window_length

    def slice_contexts(self, slice_obj):
        return self.data[slice_obj]

    def lookback(self, lookback_length: int, slide_by: int = 0):
        self._check_lb_len(lookback_length)
        max_slide = self._window_length - lookback_length
        if slide_by > max_slide:
            raise ValueError(
                f"Invalid slide_by = {slide_by} for lookback_length = {lookback_length} and Context of length = {self._window_length}. It should be 0 <= slide_by <= context_length - lookback_length"
            )

        lb_window = self.slice_contexts(slice(slide_by, lookback_length + slide_by))
        return lb_window

    def lookforward(self, lookback_length: int):
        self._check_lb_len(lookback_length)
        lf_window = self.slice_contexts(slice(lookback_length, None))
        return lf_window

    def _check_lb_len(self, lookback_length: int):
        if (lookback_length > self.context_length) or (lookback_length < 1):
            raise ValueError(
                f"Invalid lookback_length = {lookback_length} for Context of length = {self.context_length}. It should be 1 <= lookback_length <= context_length."
            )

    def __repr__(self):
        return (
            f"Contexts <context_length={self.context_length},"
            f"features={self._features_shape}>\n{self.data.__str__()}"
        )

    def save(self, path: os.PathLike):
        pickle_save(self, path)

    @classmethod
    def load(cls, filename):
        """Load a serialized Context Window from file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            ExtendedDMD: The loaded model.
        """
        return pickle_load(cls, filename)


class TrajectoryContexts(ContextWindow):
    def __init__(
        self, trajectory: np.ndarray, context_length: int = 2, time_lag: int = 1
    ):

        if context_length < 2:
            raise ValueError(f"context_length must be >= 2, got {context_length}")

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        trajectory = np.asanyarray(trajectory)
        if trajectory.ndim < 1:
            raise ShapeError(
                f"Invalid trajectory shape {trajectory.shape}. The trajectory should be at least 1D."
            )
        elif trajectory.ndim == 1:
            trajectory = trajectory[:, None]  # Add a dummy dimension for 1D features
        else:
            pass

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

        self.trajectory = trajectory
        self.time_lag = time_lag
        self.idx_map = np.moveaxis(idx_map, -1, 1)[:, ::time_lag, ...]
        self.data = np.moveaxis(data, -1, 1)[:, ::time_lag, ...]
        # Shapes
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self._num_contexts = self.shape[0]
        self._window_length = self.shape[1]
        self._features_shape = self.shape[2:]

        assert context_length == self.context_length

    def __len__(self):
        return self._num_contexts

    def slice_contexts(self, slice_obj):
        return self.data[:, slice_obj]

    def __repr__(self):
        return (
            f"Contexts <count={len(self)}, context_length={self.context_length},"
            f"features={self._features_shape}>\n{self.data.__str__()}"
        )

    def __getitem__(self, index):
        return self.data[index]


class MultiTrajectoryContexts(ContextWindow):
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

    return TrajectoryContexts(
        trajectory, context_length=context_window_len, time_lag=time_lag
    )
