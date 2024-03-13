import logging

import numpy as np
import torch

from kooplearn._src.utils import ShapeError
from kooplearn.data import TensorContextDataset

logger = logging.getLogger("kooplearn")


def collate_context_dataset(batch: list[TensorContextDataset]):
    concat_fn = torch.cat if torch.is_tensor(batch[0].data) else np.concatenate
    batched_data = torch.tensor(concat_fn([ctx.data for ctx in batch]))
    return TorchTensorContextDataset(batched_data)


class TorchTensorContextDataset(TensorContextDataset):
    def __init__(self, data: torch.Tensor):
        if data.ndim < 3:
            raise ShapeError(
                f"Invalid shape {data.shape}. The data must have be at least three dimensional [batch_size, context_len, *features]."
            )
        if not torch.is_tensor(data):
            data = torch.as_tensor(data)
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

        self.data, self.idx_map = self._build_contexts_torch(
            trajectory, context_length, time_lag
        )

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
        idx_map = (torch.arange(len(trajectory)).reshape(-1, 1)).unfold(
            0, window_shape, 1
        )

        data = torch.movedim(data, -1, 1)[:, ::time_lag, ...]
        idx_map = torch.movedim(idx_map, -1, 1)[:, ::time_lag, ...]
        return data, TensorContextDataset(idx_map)
