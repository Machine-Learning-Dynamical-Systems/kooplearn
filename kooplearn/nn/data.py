import logging

import numpy as np
import torch

from kooplearn.data import TensorContextDataset, _concatenate_contexts

logger = logging.getLogger("kooplearn")


def collate_context_dataset(batch: list[TensorContextDataset]):
    cat_data, cat_observables = _concatenate_contexts(batch)
    return TensorContextDataset(cat_data, observables=cat_observables, backend="torch")


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
    return data, idx_map
