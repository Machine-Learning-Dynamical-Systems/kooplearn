from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.check_deps import parse_backend
from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset
from copy import deepcopy


def concatenate_contexts(
    contexts: Sequence["TensorContextDataset"]
):
    """Concatenates a sequence of context windows into a single tensor.
    """
    backends = set([ctx.backend for ctx in contexts])
    context_shapes = set([tuple(ctx.shape[1:]) for ctx in contexts])
    # Check backends & context lengths are all equal
    if (len(backends) != 1) or (len(context_shapes) != 1):
        raise ValueError(
            f"All context windows must have the same backend and the same context length. Got backends: {backends}, context shapes {context_shapes}"
        )
    else:
        backend = backends.pop()
    torch, backend = parse_backend(backend)
    if backend == "numpy":
        cat_data = np.concatenate([ctx.data for ctx in contexts], axis=0)
        return TensorContextDataset(cat_data, backend=backend)
    else:
        cat_data = torch.cat([ctx.data for ctx in contexts], dim=0)
        return TensorContextDataset(cat_data, backend=backend)
          


class TensorContextDataset(ContextWindowDataset):
    """Class for a collection of context windows with tensor features."""

    def __init__(self, data: ArrayLike, observables: dict[ArrayLike] = {}, backend: str = "auto", **backend_kw):
        """Initializes the TensorContextDataset.

        Args:
            data (ArrayLike): A collection of context windows.
            observables (dict[ArrayLike], optional): A dictionary of observables. Defaults to ``{}``.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of data. Defaults to ``'auto'``.
            **backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.
        """
        # Backend selection
        torch, backend = parse_backend(backend)

        _observables = deepcopy(observables)
        _observables["__state__"] = data
        observables = {}
        observables_shapes = set()
        for obs_name, obs_data in _observables.items():
            if backend == "numpy":
                if torch is not None:
                    if torch.is_tensor(obs_data):
                        obs_data = obs_data.numpy(force=True)
                    else:
                        obs_data = np.asanyarray(obs_data, **backend_kw)
                else:
                    obs_data = np.asanyarray(obs_data, **backend_kw)
            elif backend == "torch":
                if torch is None:
                    raise ImportError(
                        "You selected the 'torch' backend, but kooplearn wasn't able to import it."
                    )
                else:
                    if torch.is_tensor(obs_data):
                        pass
                    else:
                        obs_data = torch.tensor(obs_data, **backend_kw)
            elif backend == "auto":
                if torch is not None:
                    if torch.is_tensor(obs_data):
                        backend = "torch"
                    else:
                        obs_data = np.asanyarray(obs_data, **backend_kw)
                        backend = "numpy"
                else:
                    obs_data = np.asanyarray(obs_data, **backend_kw)
                    backend = "numpy"
            

            # Attributes init
            if obs_data.ndim < 3:
                raise ShapeError(
                    f"Invalid shape for {obs_name}: {obs_data.shape}. The data must have be at least three dimensional [batch_size, "
                    f"context_len, *features]."
                )
            observables_shapes.add(obs_data.shape[:2])
            if len(observables_shapes) != 1:
                raise ValueError(
                    f"All observables must have the same context length and number of context windows. Got shapes: {observables_shapes}"
                )
            observables[obs_name] = obs_data
        
        self.data = observables.pop("__state__")
        self.observables = observables
        self.backend = backend
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self._backend_kw = backend_kw
        self._context_length = self.shape[1]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx) -> "TensorContextDataset":
        if np.issubdtype(type(idx), np.integer):
            _data = self.data[idx][None, ...]
            _obs = {obs_name: obs_data[idx][None, ...] for obs_name, obs_data in self.observables.items()}
        else:
            _data = self.data[idx]
            _obs = {obs_name: obs_data[idx] for obs_name, obs_data in self.observables.items()}
        return TensorContextDataset(_data, _obs, backend=self.backend, **self._backend_kw)

    def __getitems__(self, indices: list[int]) -> "TensorContextDataset":
        """Called by torch to index batched data directly"""
        return self.__getitem__(indices)

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
        if context_length < 1 and not isinstance(context_length, int):
            raise ValueError(
                f"context_length must be an interger >= 1, got {context_length}"
            )

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        if trajectory.ndim < 1:
            raise ShapeError(
                f"Invalid trajectory shape {trajectory.shape}. Expected at least 1D."
            )

        if multi_traj:
            if trajectory.ndim < 2:
                raise ShapeError(
                    f"Invalid trajectory shape {trajectory.shape}. Expected at least (n_trajs, n_frames, *features_shape)"
                )
        else:
            if trajectory.ndim == 1:
                trajectory = trajectory[
                    :, None
                ]  # Add a dummy dimension for 1D features

        torch, backend = parse_backend(backend)

        if multi_traj:  # Take context of each trajectory independently
            data_per_traj, idx_map_per_traj = [], []
            for traj in trajectory:
                data, idx_map = _contexts_from_traj(
                    traj,
                    context_length=context_length,
                    time_lag=time_lag,
                    backend=backend,
                    **backend_kw,
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
                trajectory,
                context_length=context_length,
                time_lag=time_lag,
                backend=backend,
                **backend_kw,
            )

        # Store reference to the original data
        self.multi_traj = multi_traj
        self.trajectory = trajectory
        self.time_lag = time_lag
        super().__init__(self.data)
        assert context_length == self.context_length


def _contexts_from_traj(
    trajectory: ArrayLike,
    context_length: int,
    time_lag: int,
    backend: str,
    **backend_kw,
) -> [np.ndarray, TensorContextDataset]:
    # Backend selection
    torch, backend = parse_backend(backend)
    if backend == "numpy":
        if torch is not None:
            if torch.is_tensor(trajectory):
                trajectory = trajectory.detach().cpu().numpy()
                data, idx_map = _contexts_from_traj_np(
                    trajectory, context_length, time_lag
                )
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                data, idx_map = _contexts_from_traj_np(
                    trajectory, context_length, time_lag
                )
        else:
            trajectory = np.asanyarray(trajectory, **backend_kw)
            data, idx_map = _contexts_from_traj_np(trajectory, context_length, time_lag)
    elif backend == "torch":
        if torch is None:
            raise ImportError(
                "You selected the 'torch' backend, but kooplearn wasn't able to import it."
            )
        else:
            from kooplearn.nn.data import _contexts_from_traj_torch

            if not torch.is_tensor(trajectory):
                trajectory = torch.tensor(trajectory, **backend_kw)
            data, idx_map = _contexts_from_traj_torch(
                trajectory, context_length, time_lag
            )
    elif backend == "auto":
        if torch is not None:
            if torch.is_tensor(trajectory):
                from kooplearn.nn.data import _contexts_from_traj_torch

                data, idx_map = _contexts_from_traj_torch(
                    trajectory, context_length, time_lag
                )
            else:
                trajectory = np.asanyarray(trajectory, **backend_kw)
                data, idx_map = _contexts_from_traj_np(
                    trajectory, context_length, time_lag
                )
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
    return data, idx_map


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
    )
