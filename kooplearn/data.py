import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.check_deps import parse_backend
from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset


class TensorContextDataset(ContextWindowDataset):
    """Class for a collection of context windows with tensor features."""

    def __init__(self, data: ArrayLike, backend: str = "auto", **backend_kw):
        """Initializes the TensorContextDataset.

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
                raise ImportError("You selected the 'torch' backend, but kooplearn wasn't able to import it.")
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
                f"Invalid shape {data.shape}. The data must have be at least three dimensional [batch_size, "
                f"context_len, *features]."
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
        else:
            try:
                return TensorContextDataset(self.data[idx])
            except Exception as e:
                raise (e)

    def slice(self, slice_obj):
        """Returns a slice of the context windows given a slice object.

        Args:
        ----
            slice_obj (slice): The python slice object.

        Returns:
        -------
            Slice of the context windows.
        """
        return self.data[:, slice_obj]

    def batched_to_flat_trajectory(self):
        """ Returns a (batch * time, *feature_dims) view of the data tensor of shape (batch, time, *feature_dims).

        This method guarantees the preservation of temporal sample ordering when reshaping the data.
        It first ensures that the tensor is contiguous in memory, which safeguards against any disruption of data order
        during the reshaping process.

        Returns:
            torch.Tensor: Reshaped view tensor of the data in the shape shape (batch * time, *feature_dims).
        """
        batch_size = len(contexts_batch)
        trail_dims = contexts_batch.shape[2:]
        x_contiguous = x.contiguous()  # Needed for reshaping not to mess with the time order.
        x_reshaped = x_contiguous.view(-1, x_contiguous.size(-1))
        return x_reshaped


class TrajectoryContextDataset(TensorContextDataset):
    """Class for a collection of context windows with tensor features."""

    def __init__(
        self,
        trajectory: ArrayLike,
        context_length: int = 2,
        time_lag: int = 1,
        multi_traj: bool = False,
        backend: str = "auto",
        **backend_kw,
    ):
        """
        Initializes the TrajectoryContextDataset. It takes as input a trajectory and returns a sequence of context windows.

        Args:
            trajectory (ArrayLike): A trajectory of shape ``(n_frames, *features_shape)`` or ``(n_trajs, n_frames, *features_shape)`` if `multi_traj` is set to ``True``.
            context_length (int, optional): Length of the context window. Default to ``2``.
            time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
            multi_traj (bool, optional): If set to ``True``, the input trajectory is intepreted as a collection of trajectories. Default to ``False``.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of the trajectory. Default to ``'auto'``.
            backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.
        """
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        if trajectory.ndim < 1:
            raise ShapeError(f"Invalid trajectory shape {trajectory.shape}. Expected at least 1D.")

        if multi_traj:
            if trajectory.ndim < 2:
                raise ShapeError(f"Invalid trajectory shape {trajectory.shape}. Expected at least (n_trajs, n_frames, *features_shape)")
        else:
            if trajectory.ndim == 1:
                trajectory = trajectory[:, None]  # Add a dummy dimension for 1D features

        torch, backend = parse_backend(backend)

        if multi_traj:  # Take context of each trajectory independently
            data_per_traj, idx_map_per_traj = [], []
            for traj in trajectory:
                data, idx_map = _contexts_from_traj(
                    traj, context_length=context_length, time_lag=time_lag, backend=backend, **backend_kw
                    )
                data_per_traj.append(data)
                idx_map_per_traj.append(idx_map)

            if isinstance(data_per_traj[0], np.ndarray):
                self.data = np.concatenate(data_per_traj, axis=0)
                self.idx_map = np.concatenate(idx_map_per_traj, axis=0)
            else:
                self.data = torch.cat(data_per_traj, dim=0)
                self.idx_map = torch.cat(idx_map_per_traj, dim=0)
        else:
            self.data, self.idx_map = _contexts_from_traj(
                trajectory, context_length=context_length, time_lag=time_lag, backend=backend, **backend_kw
                )

        # Store reference to the original data
        self.multi_traj = multi_traj
        self.trajectory = trajectory
        self.time_lag = time_lag
        super().__init__(self.data)
        assert context_length == self.context_length

def _contexts_from_traj(trajectory: ArrayLike, context_length: int, time_lag: int, backend: str, **backend_kw) -> [np.ndarray, TensorContextDataset]:
    # Backend selection
    torch, backend = parse_backend(backend)
    if backend == "numpy":
        if torch is not None:
            if torch.is_tensor(trajectory):
                trajectory = trajectory.detach().cpu().numpy()
                data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
        else:
            trajectory = np.asanyarray(trajectory, **backend_kw)
            data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
    elif backend == "torch":
        if torch is None:
            raise ImportError("You selected the 'torch' backend, but kooplearn wasn't able to import it.")
        else:
            from kooplearn.nn.data import _contexts_from_traj_torch
            if not torch.is_tensor(trajectory):
                trajectory = torch.tensor(trajectory, **backend_kw)
            data, idx_map = _contexts_from_traj_torch(trajectory, context_length, time_lag)
    elif backend == "auto":
        if torch is not None:
            if torch.is_tensor(trajectory):
                from kooplearn.nn.data import _contexts_from_traj_torch
                data, idx_map = _contexts_from_traj_torch(trajectory, context_length, time_lag)
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
        else:
            trajectory = np.asanyarray(trajectory, **backend_kw)
            data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
    return data, idx_map

def _contexts_from_traj_np(trajectory, context_length: int, time_lag: int):
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

def traj_to_contexts(
        trajectory: np.ndarray,
        context_window_len: int = 2,
        time_lag: int = 1,
        backend: str = "auto",
        **backend_kwargs,
        ):
    """Transforms a single trajectory to a sequence of context windows.

    Args:
    ----
        trajectory (np.ndarray): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Default to ``2``.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
        backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of the trajectory. Default to ``'auto'``.
        backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.

    Returns:
    -------
        TrajectoryContextDataset: A sequence of context windows.
    """
    return TrajectoryContextDataset(
        trajectory,
        context_length=context_window_len,
        time_lag=time_lag,
        multi_traj=False,
        backend=backend,
        **backend_kwargs,
        )


def multi_traj_to_context(
        trajectories: np.ndarray,
        context_window_len: int = 2,
        time_lag: int = 1,
        backend: str = "auto",
        **backend_kwargs,
        ):
    """Transforms a collection of trajectories to a sequence of context windows.

    Args:
    ----
        trajectories (np.ndarray): A trajectory of shape ``(n_trajs, n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Default to ``2``.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
        backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``,
        will use the same backend of the trajectory. Default to ``'auto'``.
        backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``,
        it is possible to specify the device of the tensor.

    Returns:
    -------
        TrajectoryContextDataset: A sequence of context windows.
    """
    return TrajectoryContextDataset(
        trajectories,
        context_length=context_window_len,
        time_lag=time_lag,
        multi_traj=True,
        backend=backend,
        **backend_kwargs,
        )

