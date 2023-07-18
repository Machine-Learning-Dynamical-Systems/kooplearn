import torch
import torch.nn as nn
# import pytorch_lightning as pl
# from dl_forecast.datamodules import AutoformerDataModule
# from dl_forecast.datasets import AutoformerDataset


class DNNModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    @classmethod
    def from_time_series_dataset(cls):
        raise NotImplementedError

    @staticmethod
    def time_series_dataset_to_model_kwargs():
        raise NotImplementedError

    @staticmethod
    def time_series_dataset_to_optimizer_kwargs():
        return {}

    @staticmethod
    def time_series_dataset_to_scheduler_kwargs():
        return {}

    @staticmethod
    def get_default_optimizer_fn():
        return torch.optim.Adam

    @staticmethod
    def get_default_optimizer_kwargs():
        return {}

    @staticmethod
    def get_default_scheduler_fn():
        return torch.optim.lr_scheduler.ExponentialLR

    @staticmethod
    def get_default_scheduler_kwargs():
        return {}

    @staticmethod
    def get_default_scheduler_config():
        return {}

    @staticmethod
    def get_default_callbacks_fns():
        return []

    @staticmethod
    def get_default_callbacks_kwargs():
        return [
            dict(),
        ]

    @staticmethod
    def get_default_loss_fn():
        return torch.nn.functional.mse_loss

    @staticmethod
    def get_default_trainer_kwargs():
        return {}

    @staticmethod
    def get_default_datamodule_class():
        return 1

    @staticmethod
    def get_default_datamodule_kwargs():
        return {}
