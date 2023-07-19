from math import ceil
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class TimeseriesDataset(Dataset):
    def __init__(self, df_series, idx_start, idx_end, lb_window_size, horizon_size, freq_date=None, is_train=True,
                 step=1, mean=None, std=None, idx_start_train=None, date_encoder_func=None):
        assert isinstance(df_series, pd.DataFrame)
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
        self.horizon_size = horizon_size
        self.is_train = is_train
        self.idx_start_train = idx_start_train
        self.step = step
        if mean is None and std is None:
            mean = self.values[idx_start:idx_end].mean(dim=0)
            std = self.values[idx_start:idx_end].std(dim=0)
        self.mean = mean
        self.std = std
        self.values = (self.values - self.mean) / self.std
        self.real_idx_start = max(self.idx_start - self.lb_window_size, 0)
        if is_train:
            self.idx_start_train = self.real_idx_start
        if self.step > 1 and not is_train:
            # we need to preserve the order, so we start before the first idx if needed...
            remainder = (self.real_idx_start - self.idx_start_train) % self.step
            if remainder > 0:
                self.real_idx_start = self.real_idx_start - remainder
        # we pad the end with 0s if needed
        remainder = ((len(self.times_idx) - self.idx_start_train) - (self.lb_window_size + self.horizon_size)) \
                    % self.step
        if remainder > 0:
            first = self.times_idx[-1] + 1
            last = first + (step - remainder)
            time_to_cat = torch.arange(first, last)
            self.times_idx = torch.cat([self.times_idx, time_to_cat])
            shape_to_cat = list(self.values.size())
            shape_to_cat[0] = len(time_to_cat)
            value_to_cat = torch.zeros(shape_to_cat)
            self.values = torch.cat([self.values, value_to_cat])
            if dates is not None:
                first_date = self.times_idx[-1] + pd.Timedelta(self.freq_date)
                last_date = first_date + pd.Timedelta(self.freq_date) * (step - remainder)
                date_to_cat = pd.date_range(first_date, last_date, freq=self.freq_date)
                dates = pd.concat([dates, pd.Series(date_to_cat)])
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
        idx_x_start = self.real_idx_start + (idx * self.step)
        idx_x_end = idx_x_start + self.lb_window_size
        idx_y_begin = idx_x_end
        idx_y_end = idx_y_begin + self.horizon_size
        x_time_idx = self.times_idx[idx_x_start:idx_x_end]
        x_value = self.values[idx_x_start:idx_x_end]
        y_time_idx = self.times_idx[idx_y_begin:idx_y_end]
        y_value = self.values[idx_y_begin:idx_y_end]
        mask_out_of_series_left = torch.zeros_like(y_value).bool()
        mask_out_of_series_right = mask_out_of_series_left
        if idx_y_begin < self.idx_start:
            mask_out_of_series_left[:self.idx_start - idx_y_begin] = True
        if idx_y_end > self.idx_end:
            mask_out_of_series_right[-(idx_y_end - self.idx_end):] = True
        # Usually we work with dimensions (..., channels, time), so we transpose values to match this convention
        data = {
            'x_time_idx': x_time_idx,
            'x_value': x_value.transpose(-1, -2),
            'y_time_idx': y_time_idx,
            'y_value': y_value.transpose(-1, -2),
            'mask_out_of_series_left': mask_out_of_series_left.transpose(-1, -2),
            'mask_out_of_series_right': mask_out_of_series_right.transpose(-1, -2),
        }
        return data

    def get_X_Y_numpy_matrices(self):
        # this is clearly not optimal, as we can construct the matrices directly shifting the original data,
        # but for the moment we do this because it's easier (no problems with padding, step, etc.)
        X = []
        Y = []
        for idx in range(len(self)):
            data = self[idx]
            # we skip the data if it's out of series (we will not use padding in the matrices)
            if data['mask_out_of_series_left'].any() or data['mask_out_of_series_right'].any():
                continue
            X.append(data['x_value'].flatten().numpy())
            Y.append(data['y_value'].flatten().numpy())
        # we should have shape (n_samples, n_total_features) = (n_samples, n_features * n_timesteps)
        X = np.stack(X)  # timesteps = lb_window_size
        Y = np.stack(Y)  # timesteps = horizon_size
        return X, Y
