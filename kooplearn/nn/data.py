import logging

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.utils import ShapeError

logger = logging.getLogger("kooplearn")
check_torch_deps()
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


class ContextsDataset(Dataset):
    """A torch ``Dataset`` of context windows. This class is also aliased in :mod:`kooplearn.data` as :class:`kooplearn.data.ContextsDataset`.

    Args:
        contexts (torch.Tensor): A tensor of context windows.
    """

    def __init__(self, contexts: torch.Tensor):
        if not isinstance(contexts, torch.Tensor):
            logger.warning(
                f"The provided contexts are of type {type(contexts)}. Converting to torch.Tensor."
            )
            contexts = torch.as_tensor(contexts)
        if contexts.ndim < 2:
            raise ShapeError(
                f"Invalid contexts shape {contexts.shape}. The contexts must be at least three dimensional [num_samples, context_len, *features]."
            )
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]


def traj_to_contexts_dataset(
    trajectory: torch.Tensor, context_window_len: int = 2, time_lag: int = 1
):
    """Convert a single trajectory to a Torch Dataset of context windows. This function is also aliased in :mod:`kooplearn.data` as :func:`kooplearn.data.traj_to_contexts_dataset`.

    Args:
        trajectory (torch.tensor): A trajectory of shape ``(n_frames, *features_shape)``.
        context_window_len (int, optional): Length of the context window. Defaults to 2.
        time_lag (int, optional): Time lag, i.e. stride, between successive context windows. Defaults to 1.

    Returns:
        An instance of :class:`kooplearn.nn.data.ContextsDataset`.
    """

    if context_window_len < 2:
        raise ValueError(f"context_window_len must be >= 2, got {context_window_len}")
    if time_lag < 1:
        raise ValueError(f"time_lag must be >= 1, got {time_lag}")
    if trajectory.ndim == 0:
        trajectory = trajectory.reshape(1, 1)
    elif trajectory.ndim == 1:
        trajectory = trajectory[:, None]

    if not isinstance(trajectory, torch.Tensor):
        logger.warning(
            f"The provided trajectory is of type {type(trajectory)}. Converting to torch.Tensor."
        )
        trajectory = torch.as_tensor(trajectory)

    _context_window_len = 1 + (context_window_len - 1) * time_lag
    if _context_window_len > trajectory.shape[0]:
        raise ValueError(
            f"Invalid combination of context_window_len={context_window_len} and time_lag={time_lag} for trajectory of length {trajectory.shape[0]}. Try reducing context_window_len or time_lag."
        )

    contexts = trajectory.unfold(0, _context_window_len, 1)
    contexts = torch.movedim(contexts, -1, 1)[:, ::time_lag, ...]
    return ContextsDataset(contexts)


class TrajToContextsDataset(ContextsDataset):
    def __init__(
        self, trajectory: torch.Tensor, context_window_len: int = 2, time_lag: int = 1
    ):
        if context_window_len < 2:
            raise ValueError(
                f"context_window_len must be >= 2, got {context_window_len}"
            )
        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")
        if trajectory.ndim == 0:
            trajectory = trajectory.reshape(1, 1)
        elif trajectory.ndim == 1:
            trajectory = trajectory[:, None]

        if not isinstance(trajectory, torch.Tensor):
            logger.warning(
                f"The provided trajectory is of type {type(trajectory)}. Converting to torch.Tensor."
            )
            trajectory = torch.as_tensor(trajectory)

        _context_window_len = 1 + (context_window_len - 1) * time_lag
        if _context_window_len > trajectory.shape[0]:
            raise ValueError(
                f"Invalid combination of context_window_len={context_window_len} and time_lag={time_lag} for trajectory of length {trajectory.shape[0]}. Try reducing context_window_len or time_lag."
            )

        self.context_len = context_window_len
        self.time_lag = time_lag
        self.contexts = trajectory.unfold(0, _context_window_len, 1)
        self.contexts = torch.movedim(self.contexts, -1, 1)[:, :: self.time_lag, ...]
