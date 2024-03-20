import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.check_deps import parse_backend
from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset


class TensorContextDataset(ContextWindowDataset):
    """
    Class for a collection of context windows with tensor features.
    """

    def __init__(self, data: ArrayLike, backend: str = "auto", **backend_kw):
        """
        Initializes the TensorContextDataset.

        Args:
            data (ArrayLike): A collection of context windows.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of data. Defaults to ``'auto'``.
            **backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.
        """
        # Backend selection
        torch, backend = parse_backend(backend)

        if backend == "numpy":
            if torch is not None:
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                else:
                    data = np.asanyarray(data, **backend_kw)
            else:
                data = np.asanyarray(data, **backend_kw)
        elif backend == "torch":
            if torch is None:
                raise ImportError(
                    "You selected the 'torch' backend, but kooplearn wasn't able to import it."
                )
            else:
                if torch.is_tensor(data):
                    pass
                else:
                    data = torch.tensor(data, **backend_kw)
        elif backend == "auto":
            if torch is not None:
                if torch.is_tensor(data):
                    pass
                else:
                    data = np.asanyarray(data, **backend_kw)
            else:
                data = np.asanyarray(data, **backend_kw)

        # Attributes init
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
            return TensorContextDataset(self.data[idx][None, ...])
        elif isinstance(idx, slice):
            return TensorContextDataset(self.data[idx])

    def slice(self, slice_obj):
        """
        Returns a slice of the context windows given a slice object.

        Args:
            slice_obj (slice): The python slice object.

        Returns:
            Slice of the context windows.
        """
        return self.data[:, slice_obj]


class TrajectoryContextDataset(TensorContextDataset):
    """
    Class for a collection of context windows with tensor features.
    """

    def __init__(
        self,
        trajectory: ArrayLike,
        context_length: int = 2,
        time_lag: int = 1,
        backend: str = "auto",
        **backend_kw,
    ):
        """
        Initializes the TrajectoryContextDataset. It takes as input a trajectory and returns a sequence of context windows.

        Args:
            trajectory (ArrayLike): A trajectory of shape ``(n_frames, *features_shape)``.
            context_length (int, optional): Length of the context window. Default to ``2``.
            time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of the trajectory. Default to ``'auto'``.
            backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.
        """
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

        # Backend selection

        torch, backend = parse_backend(backend)

        if backend == "numpy":
            if torch is not None:
                if torch.is_tensor(trajectory):
                    trajectory = trajectory.detach().cpu().numpy()
                    self.data, self.idx_map = _contexts_from_traj_np(
                        trajectory, context_length, time_lag
                    )
                else:
                    trajectory = np.asanyarray(trajectory, **backend_kw)
                    self.data, self.idx_map = _contexts_from_traj_np(
                        trajectory, context_length, time_lag
                    )
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                self.data, self.idx_map = _contexts_from_traj_np(
                    trajectory, context_length, time_lag
                )

        elif backend == "torch":
            if torch is None:
                raise ImportError(
                    "You selected the 'torch' backend, but kooplearn wasn't able to import it."
                )
            else:
                from kooplearn.nn.data import _contexts_from_traj_torch

                if not torch.is_tensor(trajectory):
                    trajectory = torch.tensor(trajectory, **backend_kw)
                self.data, self.idx_map = _contexts_from_traj_torch(
                    trajectory, context_length, time_lag
                )
        elif backend == "auto":
            if torch is not None:
                if torch.is_tensor(trajectory):
                    from kooplearn.nn.data import _contexts_from_traj_torch

                    self.data, self.idx_map = _contexts_from_traj_torch(
                        trajectory, context_length, time_lag
                    )
                else:
                    trajectory = np.asanyarray(trajectory, **backend_kw)
                    self.data, self.idx_map = _contexts_from_traj_np(
                        trajectory, context_length, time_lag
                    )
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                self.data, self.idx_map = _contexts_from_traj_np(
                    trajectory, context_length, time_lag
                )

        self.trajectory = trajectory
        self.time_lag = time_lag
        super().__init__(self.data)
        assert context_length == self.context_length


def _contexts_from_traj_np(trajectory, context_length, time_lag):
    window_shape = 1 + (context_length - 1) * time_lag
    if window_shape > trajectory.shape[0]:
        raise ValueError(
            f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of "
            f"length {trajectory.shape[0]}. Try reducing context_length or time_lag."
        )

    data = np.lib.stride_tricks.sliding_window_view(trajectory, window_shape, axis=0)

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
    **backend_kwargs,
):
    """
    Transforms a single trajectory to a sequence of context windows.

    Args:
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Default to ``2``.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
        backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of the trajectory. Default to ``'auto'``.
        backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.

    Returns:
        TrajectoryContextDataset: A sequence of context windows.
    """
    return TrajectoryContextDataset(
        trajectory,
        context_length=context_window_len,
        time_lag=time_lag,
        backend=backend,
        **backend_kwargs,
    )
