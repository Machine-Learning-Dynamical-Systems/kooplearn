import torch.functional as F
from lightning import LightningModule


class KoopmanForecasterModule(LightningModule):
    def __init__(self, model_class, model_hyperparameters, optimizer_fn, optimizer_hyperparameters,
                 koopman_estimator, koopman_estimator_hyperparameters, scheduler_fn=None,
                 scheduler_hyperparameters=None, scheduler_config=None, loss_fn=F.mse_loss):
        super().__init__()
        for k, v in model_hyperparameters.items():
            self.hparamsxz[k] = v
        for k, v in optimizer_hyperparameters.items():
            self.hparams[f'optim_{k}'] = v
        for k, v in koopman_estimator_hyperparameters.items():
            self.hparams[f'koop_{k}'] = v
        for k, v in scheduler_hyperparameters.items():
            self.hparams[f'sched_{k}'] = v
        for k, v in scheduler_config.items():
            self.hparams[f'sched_{k}'] = v
        self.save_hyperparameters()
        self.model = model_class(**model_hyperparameters)
        self.optimizer_fn = optimizer_fn
        self.koopman_estimator = koopman_estimator
        self.scheduler_fn = scheduler_fn
        self.optimizer_hyperparameters = optimizer_hyperparameters
        self.scheduler_hyperparameters = scheduler_hyperparameters
        self.scheduler_config = scheduler_config
        self.loss_fn = loss_fn

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_hyperparameters)
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, **self.scheduler_hyperparameters)
            lr_scheduler_config = self.scheduler_config.copy()
            lr_scheduler_config['scheduler'] = scheduler
            return [optimizer], [lr_scheduler_config]
        return optimizer

    def forward(self, batch):
        # dimensions convention (batch_size, channels, temporal_dim)
        return self.model(batch)

    def base_step(self, batch, batch_idx):
        # train and fit on batch
        mask_out_of_series_left = batch['mask_out_of_series_left']
        mask_out_of_series_right = batch['mask_out_of_series_right']
        y_true = batch['y_value']
        x_time = batch['x_time_idx']
        x_value = batch['x_value']
        y_time = batch['y_time_idx']
        model_output = self(batch)
        y_pred = model_output['y_pred']
        mask_out_of_series = mask_out_of_series_left + mask_out_of_series_right
        mask_in_series = ~mask_out_of_series
        y_pred = y_pred * mask_in_series
        y_true = y_true * mask_in_series
        loss = self.loss_fn(y_pred, y_true)
        outputs = {
            'loss': loss,
            'x_time_idx': x_time,
            'x_value': x_value,
            'y_time_idx': y_time,
            'y_true': y_true,
            'y_pred': y_pred,
            'mask_out_of_series': mask_out_of_series
        }
        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self.base_step(train_batch, batch_idx)
        return outputs

    def validation_step(self, valid_batch, batch_idx):
        outputs = self.base_step(valid_batch, batch_idx)
        return outputs

    def test_step(self, test_batch, batch_idx):
        outputs = self.base_step(test_batch, batch_idx)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.base_step(batch, batch_idx)
        return outputs
