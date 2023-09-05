from typing import Type, Callable

from lightning import LightningModule
import torch
from kooplearn._src.deep_learning.utils.Brunton_utils import advance_encoder_output


class BruntonModule(LightningModule):
    """Pytorch Lightning module for Brunton's model.

    Organized as a classical LightningModule check the documentation of Pytorch Lightning for more details.

    Note: We split the class/functions from the keyword arguments to be able to easily save and load the model and
    log the hyperparameters.

    Args:
        encoder_class: Class of the encoder. Can be any deep learning architecture (torch.nn.Module) that
            takes as input a dictionary containing the key 'x_value', a tensor of shape (..., n_features, temporal_dim),
            and encodes it into a tensor of shape (..., p) where p is the dimension of the autoencoder subspace.
        encoder_hyperparameters: Hyperparameters of the encoder. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the encoder.
        decoder_class: Class of the decoder. Can be any deep learning architecture (torch.nn.Module) that
            takes as input a dictionary containing the key 'x_value', a tensor of shape (..., p) where p is the
            dimension of the autoencoder subspace and decodes it into a tensor of shape
            (..., n_features * temporal_dim).
        decoder_hyperparameters: Hyperparameters of the decoder. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the decoder.
        auxiliary_network_class: Class of the auxiliary network. Can be any deep learning architecture (torch.nn.Module)
            that will be wrapped in a AuxiliaryNetworkWrapper. The auxiliary network must take as input a dictionary
            containing the key 'x_value', a tensor of shape (..., input_dim) and outputs a tensor of shape
            (..., output_dim) the auxiliary_network_class must take the keyword argument input_dim and output_dim when
            being instantiate, which will be correctly set by the AuxiliaryNetworkWrapper.
            TODO For more details, see the documentation of AuxiliaryNetworkWrapper.
        auxiliary_network_hyperparameters: Hyperparameters of the auxiliary network. Must be a dictionary containing as
            keys the names of the hyperparameters and as values the values of the hyperparameters of the auxiliary
            network. Note that the keyword arguments input_dim and output_dim will be set by the
            AuxiliaryNetworkWrapper, so they should not be included in auxiliary_network_hyperparameters.
        optimizer_fn: Optimizer function. Can be any torch.optim.Optimizer.
        optimizer_hyperparameters: Hyperparameters of the optimizer. Must be a dictionary containing as keys the names
            of the hyperparameters and as values the values of the hyperparameters of the optimizer.
        m_time_steps_linear_dynamics: Number of time steps m to enforce linear prediction, used in the linear dynamics
            loss term.
        m_time_steps_future_state_prediction: Number of time steps m to perform future state prediction, used in the
            future state prediction loss term.
        scheduler_fn: Scheduler function. Can be any torch.optim.lr_scheduler.LRScheduler.
        scheduler_hyperparameters: Hyperparameters of the scheduler. Must be a dictionary containing as keys the names
            of the hyperparameters and as values the values of the hyperparameters of the scheduler.
        scheduler_config: Configuration of the scheduler. Must be a dictionary containing as keys the names of
            the configuration parameters and as values the values of the configuration parameters of the scheduler.
            See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers for more
            information on how to configure the scheduler configuration (lr_scheduler_config in their documentation).
    """
    def __init__(
            self,
            encoder_class: Type[torch.nn.Module],
            encoder_hyperparameters: dict,
            decoder_class: Type[torch.nn.Module],
            decoder_hyperparameters: dict,
            auxiliary_network_class: Type[torch.nn.Module],
            auxiliary_network_hyperparameters: dict,
            optimizer_fn: Type[torch.optim.Optimizer], optimizer_hyperparameters: dict,
            loss_fn: Callable,
            m_time_steps_linear_dynamics: int,
            m_time_steps_future_state_prediction: int,
            scheduler_fn: Type[torch.optim.lr_scheduler.LRScheduler] = None, scheduler_hyperparameters: dict = None,
            scheduler_config: dict = None,
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

    def forward(self, x_value, t=1):
        # dimensions convention (..., channels, temporal_dim) (no number_of_consecutive_time_steps_generated)
        x_value = x_value.unsqueeze(-3)  # (..., 1, channels, temporal_dim)
        encoded_x_value_i = self.encoder({'x_value': x_value})
        decoded_advanced_encoded_x_values = []
        for _ in range(t):
            mus_omegas, lambdas = self.auxiliary_network(encoded_x_value_i.clone())
            advanced_encoded_x_value = advance_encoder_output(encoded_x_value_i, mus_omegas, lambdas)
            encoded_x_value_i = advanced_encoded_x_value.clone()
            decoded_advanced_encoded_x_value_i = self.decoder({'x_value': advanced_encoded_x_value})
            if decoded_advanced_encoded_x_value_i.shape != x_value.shape:
                # value flatten in encoder must be unflatten after decoder
                decoded_advanced_encoded_x_value_i = (decoded_advanced_encoded_x_value_i.reshape(x_value.shape))
            decoded_advanced_encoded_x_values.append(decoded_advanced_encoded_x_value_i)
        return torch.cat(decoded_advanced_encoded_x_values, dim=-3)  # (..., t, channels, temporal_dim)

    def base_step(self, batch, batch_idx):
        """Default step (train loop) used for training and validation."""
        # dimensions convention (..., number_of_consecutive_time_steps_generated, channels, temporal_dim)
        # if any y is out of series, we do not use the sample from the batch
        mask_out_of_series_left = batch['mask_out_of_series_left']
        mask_out_of_series_right = batch['mask_out_of_series_right']
        mask_out_of_series = mask_out_of_series_left | mask_out_of_series_right
        batch_out_of_series = mask_out_of_series.any(dim=-1).any(dim=-1).any(dim=-1)
        batch_in_series = ~batch_out_of_series
        x_value = batch['x_value'][batch_in_series][..., 0, :, :].unsqueeze(-3)
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
        encoded_x_value_i = encoded_x_value[..., 0, :].unsqueeze(-2)
        advanced_encoded_m_x_value = []
        decoded_advanced_encoded_m_x_value = []
        for _ in range(max(self.m_time_steps_linear_dynamics, self.m_time_steps_future_state_prediction)):
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
            advanced_encoded_m_x_value, dim=-2)[..., 0: self.m_time_steps_linear_dynamics, :]
        decoded_advanced_encoded_m_x_value = torch.cat(
            decoded_advanced_encoded_m_x_value, dim=-3)[..., 0: self.m_time_steps_future_state_prediction, :, :]
        # compute loss
        loss = self.loss_fn(
            # loss reconstruction
            decoded_encoded_x_value, x_value,
            # loss linear dynamics
            advanced_encoded_m_x_value, encoded_m_y_value[..., 0: self.m_time_steps_linear_dynamics, :],
            # loss future state prediction
            decoded_advanced_encoded_m_x_value, m_y_value[..., 0: self.m_time_steps_future_state_prediction, :, :],
        )
        outputs = {
            'loss': loss,
        }
        return outputs
