from copy import deepcopy
from math import ceil
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class TimeseriesDataset(Dataset):
    """Dataset for time series forecasting.

    Pytorch Dataset for time series forecasting. The dataset is constructed from a pandas dataframe containing the
    time series. The dataframe must contain a column 'time_idx' with the time index of the series. Optionally,
    it can contain a column 'date' with the date of the time index. If the column 'date' is present, the parameter
    freq_date must be specified in a format accepted by pandas DateOffset (
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects). Other columns are
    considered as features of the time series. By default, we encode the date as month, day, weekday and hour, but
    we can specify a function to encode the date in a different way with the parameter date_encoder_func.

    The dataset is constructed by specifying the start and end indices of the time series.

    We can define a window of time steps (lb_window_size) to add delayed features to the generated data. Generally, we
    also can define a horizon_size with the number of time steps that we want to predict, but for the Koopman operator
    we must have lb_window_size = horizon_size (a state is defined by the same number of time steps).

    We can also define a step parameter, which is the number of time steps between two consecutive samples.

    By default, the dataset is normalized by subtracting the mean and dividing by the standard deviation of each
    feature. If we are not working with a training dataset, we can specify the mean and standard deviation of the
    to normalize the data using the values from the training dataset. In this case, we must also specify the
    idx_start_train parameter, which is the start index of the training dataset, so that we can pad the data if needed,
    in order to have a number of samples that is a multiple of the step parameter, and offset the start index of the
    dataset to not change the sampling order.

    Finally, we can specify the number_of_consecutive_time_steps_generated parameter, which is the number of
    consecutive time steps that we want to generate. This is useful for the Brunton Model where we need to ensure
    linearity and prediction across multiple time steps.

    Args:
        df_series: Pandas dataframe containing the time series.
        idx_start: Start index of the time series.
        idx_end: End index of the time series.
        lb_window_size: Lookback window size of the time series.
        freq_date: Frequency of the date column.
        step: Number of time steps between two consecutive samples.
        is_train: If True, the dataset is a training dataset.
        normalize: If True, normalize the data.
        mean: Mean of the data to normalize the data.
        std: Standard deviation of the data to normalize the data.
        idx_start_train: Start index of the training dataset.
        date_encoder_func: Function to encode the date.
        number_of_consecutive_time_steps_generated: Number of consecutive time steps to generate.
    """

    # In general for koopman we must have lb_window_size = horizon_size
    def __init__(self, df_series: pd.DataFrame, idx_start: int, idx_end: int, lb_window_size: int,
                 freq_date: str = None, step: int = 1, is_train: bool = True,
                 normalize: bool = True, mean: float = None, std: float = None, idx_start_train: int = None,
                 date_encoder_func: Callable = None,
                 number_of_consecutive_time_steps_generated: int = 1):
        assert isinstance(df_series, pd.DataFrame)
        assert 'time_idx' in df_series.columns
        times = df_series['time_idx']
        if 'date' in df_series.columns:
            dates = df_series['date']
            values = df_series.drop(columns=['time_idx', 'date'])
            if freq_date is None:
                raise ValueError('freq_date must be specified if date column is present')
        else:
            dates = None
            values = df_series.drop(columns=['time_idx'])
        self.times_idx = torch.tensor(times.to_numpy(), dtype=torch.float32)
        self.values = torch.tensor(values.to_numpy(), dtype=torch.float32)
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.freq_date = freq_date
        self.is_train = is_train
        self.lb_window_size = lb_window_size
        self.horizon_size = lb_window_size
        self.is_train = is_train
        self.idx_start_train = idx_start_train
        self.step = step
        self.number_of_consecutive_time_steps_generated = number_of_consecutive_time_steps_generated
        # normalize the data if needed
        if mean is None and std is None:
            mean = self.values[idx_start:idx_end].mean(dim=0)
            std = self.values[idx_start:idx_end].std(dim=0)
        self.mean = mean
        self.std = std
        if normalize:
            self.values = (self.values - self.mean) / self.std
        self.real_idx_start = max(self.idx_start - self.lb_window_size, 0)
        if is_train:
            self.idx_start_train = self.real_idx_start
        # we offset the start index of the dataset to not change the sampling order
        if self.step > 1 and not is_train:
            # we need to preserve the order, so we start before the first idx if needed...
            remainder = (self.real_idx_start - self.idx_start_train) % self.step
            if remainder > 0:
                self.real_idx_start = self.real_idx_start - remainder
        # we pad the end with 0s if needed
        remainder = (((len(self.times_idx) - self.idx_start_train) - (self.lb_window_size + self.horizon_size))
                     % self.step)
        if remainder > 0 or number_of_consecutive_time_steps_generated > 1:
            first = self.times_idx[-1] + 1
            # when remainder is 0 and number_of_consecutive_time_steps_generated > 1 it will actually pad one more time
            # than needed, but I think it's not a big deal...
            last = first + (step - remainder) + step * (number_of_consecutive_time_steps_generated - 1)
            time_to_cat = torch.arange(first, last)
            self.times_idx = torch.cat([self.times_idx, time_to_cat])
            shape_to_cat = list(self.values.size())
            shape_to_cat[0] = len(time_to_cat)
            value_to_cat = torch.zeros(shape_to_cat)
            self.values = torch.cat([self.values, value_to_cat])
            if dates is not None:
                first_date = self.times_idx[-1] + pd.DateOffset(self.freq_date)
                last_date = (first_date + pd.DateOffset(self.freq_date) *
                             (step - remainder + step * (number_of_consecutive_time_steps_generated - 1)))
                date_to_cat = pd.date_range(first_date, last_date, freq=self.freq_date)
                dates = pd.concat([dates, pd.Series(date_to_cat)])
        # encode the date
        if date_encoder_func and dates is not None:
            dates = date_encoder_func(pd.to_datetime(dates.values), freq=self.freq_date)
        elif dates is not None:
            dates = pd.DataFrame(dates, columns=['date'])
            dates['month'] = dates['date'].dt.month
            dates['day'] = dates['date'].dt.day
            dates['weekday'] = dates['date'].dt.weekday
            dates['hour'] = dates['date'].dt.hour
            dates = dates.drop('date', axis=1)
        if dates is not None:
            self.dates = torch.tensor(dates.to_numpy(), dtype=torch.float32)
        else:
            self.dates = None

    def __len__(self):
        return ceil(((self.idx_end - self.real_idx_start) - (self.lb_window_size + self.horizon_size)) / self.step + 1)

    def __getitem__(self, idx):
        """Get item.

        Get the data for the idx-th sample. The data is a dictionary with the following keys:
            - x_time_idx: time index of the input data
            - x_value: input data
            - y_time_idx: time index of the output data
            - y_value: output data
            - mask_out_of_series_left: mask indicating if the data is out of series on the left
            - mask_out_of_series_right: mask indicating if the data is out of series on the right

        The shapes of the tensors are (features, lb_window_size) for the case where we generate only one time step, and
        (number_of_consecutive_time_steps_generated, features, lb_window_size) for the case where we generate more than
        one time step.
        """
        xs_time_idx = []
        xs_value = []
        ys_time_idx = []
        ys_value = []
        masks_out_of_series_left = []
        masks_out_of_series_right = []
        for i in range(self.number_of_consecutive_time_steps_generated):
            idx_x_start = self.real_idx_start + ((idx + i) * self.step)
            idx_x_end = idx_x_start + self.lb_window_size
            idx_y_begin = idx_x_end
            idx_y_end = idx_y_begin + self.horizon_size
            x_time_idx = self.times_idx[idx_x_start:idx_x_end]
            x_value = self.values[idx_x_start:idx_x_end]
            y_time_idx = self.times_idx[idx_y_begin:idx_y_end]
            y_value = self.values[idx_y_begin:idx_y_end]
            mask_out_of_series_left = torch.zeros_like(y_value).bool()
            mask_out_of_series_right = deepcopy(mask_out_of_series_left)
            if idx_y_begin < self.idx_start:
                mask_out_of_series_left[:self.idx_start - idx_y_begin] = True
            if idx_y_end > self.idx_end:
                mask_out_of_series_right[-(idx_y_end - self.idx_end):] = True
            # Usually we work with dimensions (..., channels, time), so we transpose values to match this convention
            xs_time_idx.append(x_time_idx)
            xs_value.append(x_value.transpose(-1, -2))
            ys_time_idx.append(y_time_idx)
            ys_value.append(y_value.transpose(-1, -2))
            masks_out_of_series_left.append(mask_out_of_series_left.transpose(-1, -2))
            masks_out_of_series_right.append(mask_out_of_series_right.transpose(-1, -2))
        # Note that for the case where we generate more than one time step, we stack the tensors, and we get shapes
        # (number_of_consecutive_time_steps_generated, channels, time)
        # For the case where we generate only one time step, we get the shapes (channels, time)
        data = {
            'x_time_idx': torch.stack(xs_time_idx).squeeze(dim=0),
            'x_value': torch.stack(xs_value).squeeze(dim=0),
            'y_time_idx': torch.stack(ys_time_idx).squeeze(dim=0),
            'y_value': torch.stack(ys_value).squeeze(dim=0),
            'mask_out_of_series_left': torch.stack(masks_out_of_series_left).squeeze(dim=0),
            'mask_out_of_series_right': torch.stack(masks_out_of_series_right).squeeze(dim=0),
        }
        return data

    def get_numpy_matrices(self):
        """Get numpy matrices of input and output data.

        Get the X and Y matrices, where X is the input data and Y is the output data. The matrices are numpy arrays with
        the entire dataset, so they can be used for training with other frameworks.

        Returns:
            X: numpy array with the input data
            Y: numpy array with the output data
        """
        # this is clearly not optimal, as we can construct the matrices directly shifting the original data,
        # but for the moment we do this because it's easier (no problems with padding, step, etc.)
        X = []
        Y = []
        # We temporarily change the number of consecutive time steps generated to 1, so that we can iterate over the
        # data without generating multiple time steps
        save_number_of_consecutive_time_steps_generated = self.number_of_consecutive_time_steps_generated
        self.number_of_consecutive_time_steps_generated = 1
        for idx in range(len(self)):
            data = self[idx]
            # we skip the data if it's out of series (we will not use padding in the matrices)
            if data['mask_out_of_series_left'].any() or data['mask_out_of_series_right'].any():
                continue
            X.append(data['x_value'].flatten().numpy())
            Y.append(data['y_value'].flatten().numpy())
        self.number_of_consecutive_time_steps_generated = save_number_of_consecutive_time_steps_generated
        # we should have shape (n_samples, n_total_features) = (n_samples, n_features * n_timesteps)
        X = np.stack(X)  # timesteps = lb_window_size
        Y = np.stack(Y)  # timesteps = horizon_size
        return X, Y
