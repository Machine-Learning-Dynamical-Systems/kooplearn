#Data loading and transformation
"""
Needed features:

- Data transformations (e.g. standardizing, but also common MD transformations such as alignment)
- Windowing
- Batching and data loading (check wether using TF or PyTorch dataloaders)
"""

#Imports

import numpy as np
import jax.numpy as jnp
import sklearn.preprocessing #use the functions scale or robust_scale to standardize
from torch.utils import data

#Time lagged data rearranging
def time_lagged_rearrange(data: np.ndarray, max_lag: int) -> jnp.ndarray:
    """_summary_

    Args:
        data (np.ndarray): A trajectory of shape [num_points, num_features]
        max_lag (int): Maximum time lag to evaluate

    Returns:
        np.ndarray: A rearranged dataset of shape [max_lag + 1, None, num_features]. If we let x_i(0) := [0,  i, ...] one has x_i(t) := [t, i ...]
    """
    max_idx = data.shape[0] - 1
    N = max_idx//max_lag
    base_ids = np.arange(N, dtype=int)*max_lag
    rearranged_data = np.zeros((max_lag + 1, base_ids.shape[0], data.shape[1]), dtype=data.dtype)
    for i in range(max_lag + 1):
        rearranged_data[i,...] = data[base_ids + i]
    return jnp.asarray(rearranged_data) #[num_lagtimes, num_windows, num_in_features]

class TimeLaggedDataset(data.Dataset):
    def __init__(self, data: np.ndarray, max_lag: int = 1, scale: bool = True):
        super().__init__()
        if scale:
            data = sklearn.preprocessing.robust_scale(data)
        self._time_lagged_data = time_lagged_rearrange(data, max_lag)        
        self._data_len = self._time_lagged_data.shape[1]

    def __getitem__(self, index):
        return self._time_lagged_data[:,index,:]

    def __len__(self):
        return self._data_len

def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)
