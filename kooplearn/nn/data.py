import logging

import numpy as np
import torch

from kooplearn.data import TensorContextDataset

logger = logging.getLogger("kooplearn")


def collate_context_dataset(batch: list[TensorContextDataset]):
    concat_fn = torch.cat if torch.is_tensor(batch[0].data) else np.concatenate
    batched_data = torch.tensor(concat_fn([ctx.data for ctx in batch]))
    return TensorContextDataset(batched_data)


def _contexts_from_traj_torch(trajectory, context_length, time_lag):
    window_shape = 1 + (context_length - 1) * time_lag
    if window_shape > trajectory.shape[0]:
        raise ValueError(
            f"Invalid combination of context_length={context_length} and time_lag={time_lag} for trajectory of "
            f"length {trajectory.shape[0]}. Try reducing context_length or time_lag."
        )

    data = trajectory.unfold(0, window_shape, 1)
    idx_map = (torch.arange(len(trajectory)).reshape(-1, 1)).unfold(0, window_shape, 1)

    data = torch.movedim(data, -1, 1)[:, ::time_lag, ...]
    idx_map = torch.movedim(idx_map, -1, 1)[:, ::time_lag, ...]
    return data, TensorContextDataset(idx_map)
