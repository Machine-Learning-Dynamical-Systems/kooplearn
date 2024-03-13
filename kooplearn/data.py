import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset


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

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        if np.issubdtype(type(idx), np.integer):
            return TensorContextDataset(self.data[slice(idx, idx + 1)])
        else:
            return TensorContextDataset(self.data[idx])

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

        if trajectory.ndim < 1:
            raise ShapeError(
                f"Invalid trajectory shape {trajectory.shape}. The trajectory should be at least 1D."
            )
        elif trajectory.ndim == 1:
            trajectory = trajectory[:, None]  # Add a dummy dimension for 1D features
        else:
            pass

        # It should be converted to Numpy
        trajectory = np.asanyarray(trajectory)
        self.data, self.idx_map = self._build_contexts_np(
            trajectory, context_length, time_lag
        )

        self.trajectory = trajectory
        self.time_lag = time_lag
        super().__init__(self.data)
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
            np.arange(trajectory.shape[0], dtype=np.int_).reshape(-1, 1),
            window_shape,
            axis=0,
        )

        idx_map = np.moveaxis(idx_map, -1, 1)[:, ::time_lag, ...]
        data = np.moveaxis(data, -1, 1)[:, ::time_lag, ...]
        return data, TensorContextDataset(idx_map)


class MultiTrajectoryContextDataset(TrajectoryContextDataset):
    def __init__(self):
        raise NotImplementedError


def traj_to_contexts(
    trajectory: np.ndarray,
    context_window_len: int = 2,
    time_lag: int = 1,
    backend: str = "auto",
):
    """Convert a single trajectory to a sequence of context windows.

    Args:
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Defaults to 1.

    Returns:
        TrajectoryContexts: A sequence of Context Windows.
    """
    if backend not in ["auto", "numpy", "torch"]:
        raise ValueError(
            f"Invalid backend {backend}. Accepted values are 'auto', 'numpy', or 'torch'."
        )
    # Check if torch is available
    try:
        import torch
    except ImportError:
        torch = None

    if backend == "numpy":
        if torch is not None:
            if torch.is_tensor(trajectory):
                trajectory = trajectory.detach().cpu().numpy()
        return TrajectoryContextDataset(
            trajectory, context_length=context_window_len, time_lag=time_lag
        )
    elif backend == "torch":
        try:
            from kooplearn.nn.data import TorchTrajectoryContextDataset

            return TorchTrajectoryContextDataset(
                trajectory, context_length=context_window_len, time_lag=time_lag
            )
        except ImportError:
            raise
    elif backend == "auto":
        if torch is not None:
            if torch.is_tensor(trajectory):
                from kooplearn.nn.data import TorchTrajectoryContextDataset

                return TorchTrajectoryContextDataset(
                    trajectory, context_length=context_window_len, time_lag=time_lag
                )
        return TrajectoryContextDataset(
            trajectory, context_length=context_window_len, time_lag=time_lag
        )
