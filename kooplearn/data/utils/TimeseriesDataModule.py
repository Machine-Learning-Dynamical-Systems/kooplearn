import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from kooplearn.Datasets.TimeseriesDataset import TimeseriesDataset
from torch.utils.data import Dataset

class EmptyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(self, df_series, n_train, n_valid, n_test, lb_window_size, horizon_size, batch_size=1, step=1,
                 num_workers=0, dataset_name=None):
        super().__init__()
        self.save_hyperparameters(ignore=['df_series'])
        assert isinstance(df_series, pd.DataFrame)
        self.df_series = df_series
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lb_window_size = lb_window_size
        self.horizon_size = horizon_size
        self.step = step
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.mean = None
        self.std = None
        self.idx_start_train = None

    def setup(self, stage):
        dataset_class = TimeseriesDataset
        if stage == "fit":
            idx_start_train = len(self.df_series) - self.n_valid - self.n_test - self.n_train
            idx_end_train = len(self.df_series) - self.n_valid - self.n_test
            idx_start_valid = idx_end_train
            idx_end_valid = idx_start_valid + self.n_valid
            self.train_dataset = dataset_class(
                df_series=self.df_series,
                idx_start=idx_start_train,
                idx_end=idx_end_train,
                lb_window_size=self.lb_window_size,
                horizon_size=self.horizon_size,
                is_train=True,
                step=self.step,
                mean=None,
                std=None,
                idx_start_train=None
            )
            self.mean = self.train_dataset.mean
            self.std = self.train_dataset.std
            self.idx_start_train = self.train_dataset.real_idx_start
            if self.n_valid > 0:
                self.valid_dataset = dataset_class(
                    df_series=self.df_series,
                    idx_start=idx_start_valid,
                    idx_end=idx_end_valid,
                    lb_window_size=self.lb_window_size,
                    horizon_size=self.horizon_size,
                    is_train=False,
                    step=self.step,
                    mean=self.mean,
                    std=self.std,
                    idx_start_train=self.idx_start_train
                )
            else:
                self.valid_dataset = None
        elif stage == 'test':
            if self.n_test > 0:
                idx_start_test = len(self.df_series) - self.n_test
                idx_end_test = idx_start_test + self.n_test
                self.test_dataset = dataset_class(
                    df_series=self.df_series,
                    idx_start=idx_start_test,
                    idx_end=idx_end_test,
                    lb_window_size=self.lb_window_size,
                    horizon_size=self.horizon_size,
                    is_train=False,
                    step=self.step,
                    mean=self.mean,
                    std=self.std,
                    idx_start_train=self.idx_start_train
                )
            else:
                self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        # Maybe we can just disable with limit_val_batches=0
        if self.n_valid > 0:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, drop_last=False)
        else:
            return DataLoader(EmptyDataset(), batch_size=2, drop_last=True)

    def test_dataloader(self):
        if self.n_test > 0:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, drop_last=False)
        else:
            return DataLoader(EmptyDataset(), batch_size=2, drop_last=True)
