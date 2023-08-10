from lightning import LightningModule
import torch
from kooplearn._src.deep_learning.utils.Brunton_utils import advance_encoder_output


class DPNetModule(LightningModule):
    def __init__(self,
                 encoder_class,
                 encoder_hyperparameters,
                 decoder_class,
                 decoder_hyperparameters,
                 auxiliary_network_class,
                 auxiliary_network_hyperparameters,
                 optimizer_fn,
                 optimizer_hyperparameters,
                 loss_fn,
                 m_time_steps_linear_dynamics,
                 m_time_steps_future_state_prediction,
                 scheduler_fn=None,
                 scheduler_hyperparameters=None,
                 scheduler_config=None,
                 ):
        super().__init__()
        for k, v in encoder_hyperparameters.items():
            self.hparams[f'encoder_{k}'] = v
        for k, v in decoder_hyperparameters.items():
            self.hparams[f'decoder_{k}'] = v
        for k, v in auxiliary_network_hyperparameters.items():
            self.hparams[f'auxiliary_network_{k}'] = v
        for k, v in optimizer_hyperparameters.items():
            self.hparams[f'optim_{k}'] = v
        for k, v in scheduler_hyperparameters.items():
            self.hparams[f'sched_{k}'] = v
        for k, v in scheduler_config.items():
            self.hparams[f'sched_{k}'] = v
        self.save_hyperparameters()
        self.encoder = encoder_class(**encoder_hyperparameters)
        self.decoder = decoder_class(**decoder_hyperparameters)
        self.auxiliary_network = auxiliary_network_class(**auxiliary_network_hyperparameters)
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.optimizer_hyperparameters = optimizer_hyperparameters
        self.scheduler_hyperparameters = scheduler_hyperparameters
        self.scheduler_config = scheduler_config
        self.loss_fn = loss_fn
        self.m_time_steps_linear_dynamics = m_time_steps_linear_dynamics
        self.m_time_steps_future_state_prediction = m_time_steps_future_state_prediction

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_hyperparameters)
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, **self.scheduler_hyperparameters)
            lr_scheduler_config = self.scheduler_config.copy()
            lr_scheduler_config['scheduler'] = scheduler
            return [optimizer], [lr_scheduler_config]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        outputs = self.base_step(train_batch, batch_idx)
        return outputs

    def validation_step(self, valid_batch, batch_idx):
        outputs = self.base_step(valid_batch, batch_idx)
        return outputs

    def forward(self, batch):
        # dimensions convention (..., number_of_consecutive_time_steps_generated, channels, temporal_dim)
        encoded_x_value = self.encoder(batch)
        mus_omegas, lambdas = self.auxiliary_network(encoded_x_value)
        advanced_encoded_x_value = advance_encoder_output(encoded_x_value, mus_omegas, lambdas)
        return self.decoder({'x_value': advanced_encoded_x_value})

    def base_step(self, batch, batch_idx):
        # dimensions convention (..., number_of_consecutive_time_steps_generated, channels, temporal_dim)
        # if any y is out of series, we do not use the sample from the batch
        mask_out_of_series_left = batch['mask_out_of_series_left']
        mask_out_of_series_right = batch['mask_out_of_series_right']
        mask_out_of_series = mask_out_of_series_left | mask_out_of_series_right
        batch_out_of_series = mask_out_of_series.any(dim=-1).any(dim=-1).any(dim=-1)
        batch_in_series = ~batch_out_of_series
        x_value = batch['x_value'][batch_in_series][..., 0, :, :]
        m_y_value = batch['y_value'][batch_in_series]
        # compute what is needed for reconstruction loss (decoded_encoded_x_value)
        # we only get the first time_step_generated to avoid computing the loss m times for each sample
        encoded_x_value = self.encoder({'x_value': x_value.clone()})
        decoded_encoded_x_value = self.decoder({'x_value': encoded_x_value.clone()})
        if decoded_encoded_x_value.shape != x_value.shape:
            # value flatten in encoder must be unflatten after decoder
            decoded_encoded_x_value = decoded_encoded_x_value.reshape(x_value.shape)
        # compute what is needed for linear dynamics loss (advanced_encoded_x_value, encoded_m_y_value) and
        # what is needed for future state prediction loss (decoded_advanced_encoded_x_value)
        encoded_m_y_value = self.encoder({'x_value': m_y_value.clone()})
        encoded_x_value_i = encoded_x_value[..., 0, :]
        advanced_encoded_m_x_value = []
        decoded_advanced_encoded_m_x_value = []
        for _ in max(self.m_time_steps_linear_dynamics, self.m_time_steps_future_state_prediction):
            mus_omegas, lambdas = self.auxiliary_network(encoded_x_value_i.clone())
            advanced_encoded_x_value_i = advance_encoder_output(encoded_x_value_i, mus_omegas, lambdas)
            advanced_encoded_m_x_value.append(advanced_encoded_x_value_i.clone())
            encoded_x_value_i = advanced_encoded_x_value_i.clone()
            decoded_advanced_encoded_x_value_i = self.decoder({'x_value': advanced_encoded_x_value_i})
            if decoded_advanced_encoded_x_value_i.shape != x_value.shape:
                # value flatten in encoder must be unflatten after decoder
                decoded_advanced_encoded_x_value_i = (decoded_advanced_encoded_x_value_i.reshape(x_value.shape))
            decoded_advanced_encoded_m_x_value.append(decoded_advanced_encoded_x_value_i)
        advanced_encoded_m_x_value = torch.cat(
            advanced_encoded_m_x_value, dim=-3)[..., 0: self.m_time_steps_linear_dynamics, :, :]
        decoded_advanced_encoded_m_x_value = torch.cat(
            decoded_advanced_encoded_m_x_value, dim=-3)[..., 0: self.m_time_steps_future_state_prediction, :, :]
        # compute loss
        loss = self.loss_fn(
            decoded_encoded_x_value, x_value,  # loss reconstruction
            advanced_encoded_m_x_value, encoded_m_y_value,  # loss linear dynamics
            decoded_advanced_encoded_m_x_value, m_y_value,  # loss future state prediction
        )
        outputs = {
            'loss': loss,
        }
        return outputs
