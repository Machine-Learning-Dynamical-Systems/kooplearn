import logging

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.utils import ShapeError

logger = logging.getLogger("kooplearn")
check_torch_deps()
import torch  # noqa: E402
from kooplearn.data import TensorContextDataset

class TorchTensorContextDataset(TensorContextDataset):
    def __init__(self, data: torch.Tensor):
        if data.ndim < 3:
            raise ShapeError(
                f"Invalid shape {data.shape}. The data must have be at least three dimensional [batch_size, context_len, *features]."
            )
        super().__init__(data)

class TorchTrajectoryContextDataset(TorchTensorContextDataset):
    def __init__(
        self, trajectory: torch.Tensor, context_length: int = 2, time_lag: int = 1
    ):
        if not isinstance(trajectory, torch.Tensor):
            logger.warning(
                f"The provided contexts are of type {type(trajectory)}. Converting to torch.Tensor."
            )
            trajectory = torch.as_tensor(trajectory)

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

        self.data, self.idx_map = self._build_contexts_torch(trajectory, context_length, time_lag)

        self.trajectory = trajectory
        self.time_lag = time_lag
        super().__init__(self.data)
        assert context_length == self.context_length

    def _build_contexts_torch(self, trajectory, context_length, time_lag):
        window_shape = 1 + (context_length - 1) * time_lag
        if window_shape > trajectory.shape[0]:
            raise ValueError(
                f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of "
                f"length {trajectory.shape[0]}. Try reducing context_length or time_lag."
            )

        data = trajectory.unfold(0, window_shape, 1)
        idx_map = torch.arange(len(trajectory)).unfold(0, window_shape, 1)

        data = torch.movedim(data, -1, 1)[:, ::time_lag, ...]
        idx_map = torch.movedim(idx_map, -1, 1)[:, ::time_lag, ...]
        return data, idx_map


def torch_traj_to_contexts(
    trajectory: torch.Tensor,
    context_window_len: int = 2,
    time_lag: int = 1,
):
    """Convert a single trajectory to a sequence of context windows.

    Args:
        trajectory (torch.Tensor): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Defaults to 1.

    Returns:
        TrajectoryContexts: A sequence of Context Windows.
    """

    return TorchTrajectoryContextDataset(
        trajectory, context_length=context_window_len, time_lag=time_lag
    )