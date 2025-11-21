from copy import deepcopy
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.check_deps import parse_backend
from kooplearn._src.utils import ShapeError
from kooplearn.abc import ContextWindowDataset


def _concatenate_contexts(
    contexts: Sequence["TensorContextDataset"], backend: str = None
):
    """Concatenates a sequence of context windows into a single tensor."""
    if backend is None:
        backends = set([ctx.backend for ctx in contexts])
    else:
        backends = set([backend])
    context_shapes = set([tuple(ctx.shape[1:]) for ctx in contexts])
    # Check backends & context lengths are all equal
    if (len(backends) != 1) or (len(context_shapes) != 1):
        raise ValueError(
            f"All context windows must have the same backend and the same context length. Got backends: {backends}, context shapes {context_shapes}"
        )
    else:
        for obs in [ctx.observables for ctx in contexts]:
            if set(obs.keys()) != set(contexts[0].observables.keys()):
                raise ValueError(
                    f"Observables must have the same keys for all trajectories. "
                    f"Got {set(obs.keys())} for the first trajectory, and {set(contexts[0].observables.keys())} for the second."
                )
        backend = backends.pop()
    torch, backend = parse_backend(backend)
    if backend == "numpy":
        cat_data = np.concatenate([ctx.data for ctx in contexts], axis=0)
        cat_observables = {}
        for obs_name in contexts[0].observables.keys():
            cat_observables[obs_name] = np.concatenate(
                [ctx.observables[obs_name] for ctx in contexts], axis=0
            )
        return cat_data, cat_observables
    else:
        cat_data = torch.cat([torch.tensor(ctx.data) for ctx in contexts], dim=0)
        cat_observables = {}
        for obs_name in contexts[0].observables.keys():
            cat_observables[obs_name] = torch.cat(
                [torch.tensor(ctx.observables[obs_name]) for ctx in contexts], dim=0
            )
        return cat_data, cat_observables


class TensorContextDataset(ContextWindowDataset):
    """Class for a collection of context windows with tensor features."""

    def __init__(
        self,
        data: ArrayLike,
        observables: dict[ArrayLike] = {},
        backend: str = "auto",
        **backend_kw,
    ):
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
            _obs = {
                obs_name: obs_data[idx][None, ...]
                for obs_name, obs_data in self.observables.items()
            }
        else:
            _data = self.data[idx]
            _obs = {
                obs_name: obs_data[idx]
                for obs_name, obs_data in self.observables.items()
            }
        return TensorContextDataset(
            _data, _obs, backend=self.backend, **self._backend_kw
        )

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
        observables: dict[ArrayLike] = {},
        context_length: int = 2,
        time_lag: int = 1,
        backend: str = "auto",
        **backend_kw,
    ):
        """
        Initializes the TrajectoryContextDataset. It takes as input a trajectory and returns a sequence of context windows.

        Args:
            trajectory (ArrayLike): A trajectory of shape ``(n_frames, *features_shape)``.
            observables (dict[ArrayLike], optional): A dictionary of observables. Defaults to ``{}``.
            context_length (int, optional): Length of the context window. Default to ``2``.
            time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend of the trajectory. Default to ``'auto'``.
            backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``, it is possible to specify the device of the tensor.
        """
        if context_length < 1 and not isinstance(context_length, int):
            raise ValueError(
                f"context_length must be an interger >= 1, got {context_length}"
            )

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        _observables = deepcopy(observables)
        _observables["__state__"] = trajectory
        observables = {}
        trajectory_lengths = set()
        _, backend = parse_backend(backend)
        for obs_name, obs_data in _observables.items():
            if obs_data.ndim < 1:
                raise ShapeError(
                    f"Invalid trajectory for {obs_name} shape {obs_data.shape}. Expected at least 1D."
                )

            if obs_data.ndim == 1:
                obs_data = obs_data[:, None]  # Add a dummy dimension for 1D features

            trajectory_lengths.add(obs_data.shape[0])
            if len(trajectory_lengths) != 1:
                raise ValueError(
                    f"All observables must have the same number of frames. Got lengths: {trajectory_lengths}"
                )
            ctx_data, idx_map = _contexts_from_traj(
                obs_data,
                context_length=context_length,
                time_lag=time_lag,
                backend=backend,
                **backend_kw,
            )
            observables[obs_name] = ctx_data
            if obs_name == "__state__":
                self.__idxmap__ = idx_map

        data = observables.pop("__state__")
        self.time_lag = time_lag
        super().__init__(data, observables, backend=backend, **backend_kw)
        assert context_length == self.context_length

    def future(self, lookback_length: int, time_steps: int = 1):
        """
        Returns the lookforward window of the context window after ``time_steps`` steps in the future.

        Args:
            lookback_length (int): Length of the lookback window.
            time_steps (int): Number of steps to look forward in the future. Defaults to 1.

        Returns:
            Lookforward window of the context window evolved by ``time_steps``. Out of bounds values will be filled with ``nan``.
        """
        if self.time_lag != 1:
            raise NotImplementedError()
        lf_window = self.lookforward(lookback_length)
        if time_steps == 1:
            return lf_window
        elif (time_steps > 0) and (time_steps <= lf_window.shape[0]):
            torch, backend = parse_backend(self.backend)
            if backend == "numpy":
                lf_window = np.roll(lf_window, 1 - time_steps, axis=0)
                lf_window[1 - time_steps :] = np.nan
            else:
                lf_window = torch.roll(lf_window, 1 - time_steps, dims=0)
                lf_window[1 - time_steps :] = torch.nan
            return lf_window
        else:
            raise ValueError(
                f"Invalid number of {time_steps=}. Time steps should be greater than 0 and smaller than {len(self)=}"
            )


class MultiTrajectoryContextDataset(TensorContextDataset):
    def __init__(
        self,
        trajectories: Sequence[ArrayLike],
        observables: Sequence[dict] = None,
        context_length: int = 2,
        time_lag: int = 1,
        backend: str = "auto",
        **backend_kwargs,
    ):
        """Transforms a collection of trajectories to a sequence of context windows.

        Args:
        ----
            trajectories (np.ndarray): A sequence of trajectories of shape ``(n_frames, *features_shape)``.
            observables (dict[ArrayLike], optional): A dictionary of observables. Defaults to ``None``. If passed, must be a sequence of dictionaries with the same length as ``trajectories``, and with the same observables for each trajectory.
            context_window_len (int, optional): Length of the context window. Default to ``2``.
            time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Default to ``1``.
            backend (str, optional): Specifies the backend to be used (``'numpy'``, ``'torch'``). If set to ``'auto'``,
            will use the same backend of the trajectory. Default to ``'auto'``.
            backend_kw (dict, optional): Keyword arguments to pass to the backend. For example, if ``'torch'``,
            it is possible to specify the device of the tensor.

        Returns:
        -------
            MultiTrajectoryContextDataset: A sequence of context windows.
        """
        if observables is not None:
            if len(observables) != len(trajectories):
                raise ValueError(
                    f"The number of observables ({len(observables)}) must be equal to the number of trajectories ({len(trajectories)})."
                )
            # Check that each dict has the same keys
            for obs in observables:
                if set(obs.keys()) != set(observables[0].keys()):
                    raise ValueError(
                        f"Observables must have the same keys for all trajectories. "
                        f"Got {set(obs.keys())} for the first trajectory, and {set(observables[0].keys())} for the second."
                    )
        else:
            observables = [{} for _ in range(len(trajectories))]

        ctx_list = []
        for traj, obs in zip(trajectories, observables):
            ctx_list.append(
                TrajectoryContextDataset(
                    traj,
                    observables=obs,
                    context_length=context_length,
                    time_lag=time_lag,
                    backend=backend,
                    **backend_kwargs,
                )
            )
        cat_data, cat_observables = _concatenate_contexts(ctx_list)
        self.time_lag = time_lag
        self.__idxmap__ = [ctx.__idxmap__ for ctx in ctx_list]
        self.__ctx_lengths__ = [len(ctx) for ctx in ctx_list]
        super().__init__(
            cat_data, observables=cat_observables, backend=backend, **backend_kwargs
        )

    def unstack(self):
        raise NotImplementedError()

    def future(self, lookback_length: int, time_steps: int = 1):
        """
        Returns the lookforward window of the context window after ``time_steps`` steps in the future.

        Args:
            lookback_length (int): Length of the lookback window.
            time_steps (int): Number of steps to look forward in the future. Defaults to 1.

        Returns:
            Lookforward window of the context window evolved by ``time_steps``. Out of bounds values will be filled with ``nan``.
        """
        if self.time_lag != 1:
            raise NotImplementedError()
        lf_window = self.lookforward(lookback_length)
        if time_steps == 1:
            return lf_window

        elif (time_steps > 0) and (time_steps <= max(self.__ctx_lengths__)):
            torch, backend = parse_backend(self.backend)
            if backend == "numpy":
                sections = np.cumsum(self.__ctx_lengths__)
                assert sections[-1] == len(lf_window)
                splits = np.split(lf_window, sections[:-1])
                rolled_splits = []
                for split in splits:
                    split = np.roll(split, 1 - time_steps, axis=0)
                    split[1 - time_steps :] = np.nan
                    rolled_splits.append(split)
                lf_window = np.concatenate(rolled_splits, axis=0)
            else:
                sections = torch.cumsum(torch.tensor(self.__ctx_lengths__), dim=0)
                assert sections[-1].item() == len(lf_window)
                splits = torch.tensor_split(lf_window, sections[:-1], dim=0)
                rolled_splits = []
                for split in splits:
                    split = torch.roll(split, 1 - time_steps, dims=0)
                    split[1 - time_steps :] = torch.nan
                    rolled_splits.append(split)
                lf_window = torch.cat(rolled_splits, dim=0)
            return lf_window
        else:
            raise ValueError(
                f"Invalid number of {time_steps=}. Time steps should be greater than 0 and smaller than {len(self)=}"
            )


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
    trajectory: ArrayLike,
    observables: dict = {},
    context_window_len: int = 2,
    time_lag: int = 1,
    backend: str = "auto",
    **backend_kwargs,
):
    """Transforms a single trajectory to a sequence of context windows.

    Args:
    ----
        trajectory (ArrayLike): A trajectory of shape ``(n_frames, *features_shape)``.
        observables (dict[ArrayLike], optional): A dictionary of observables. Defaults to ``{}``.
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
        observables=observables,
        context_length=context_window_len,
        time_lag=time_lag,
        backend=backend,
        **backend_kwargs,
    )


def multi_traj_to_context(
    trajectories: Sequence[ArrayLike],
    observables: Sequence[dict] = None,
    context_window_len: int = 2,
    time_lag: int = 1,
    backend: str = "auto",
    **backend_kwargs,
):
    """Transforms a collection of trajectories to a sequence of context windows.

    Args:
    ----
        trajectories (np.ndarray): A sequence of trajectories of shape ``(n_frames, *features_shape)``.
        observables (dict[ArrayLike], optional): A dictionary of observables. Defaults to ``None``. If passed, must be a sequence of dictionaries with the same length as ``trajectories``, and with the same observables for each trajectory.
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
    return MultiTrajectoryContextDataset(
        trajectories,
        observables=observables,
        context_length=context_window_len,
        time_lag=time_lag,
        backend=backend,
        **backend_kwargs,
    )
