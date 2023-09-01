from typing import Type
import torch
from lightning import LightningModule
from torch import nn
from kooplearn.models.feature_maps.DPNets.loss_and_reg_fns import projection_loss, relaxed_projection_loss, fro_reg, von_neumann_reg, log_fro_reg

class DPNetsLightningModule(LightningModule):
    """Pytorch Lightning module for DPNet feature map.

    Organized as a standard LightningModule. 

    Note: We split the class/functions from the keyword arguments to be able to easily save and load the model and
    log the hyperparameters.

    Parameters:
        encoder (torch.nn.Module): Torch module used as data encoder. Can be any ``torch.nn.Module`` taking as input a tensor of shape ``(n_samples, ...)`` and returning a *two-dimensional* tensor of shape ``(n_samples, encoded_dimension)``.
        encoder_kwargs (dict): Hyperparameters used for initializing the encoder.
        weight_sharing (bool): Whether to share the weights between the encoder of the initial data and the encoder of the evolved data. As reported in :footcite:`Kostic2023DPNets`, this can be safely set to ``True`` when the dynamical system is time-reversal invariant. Defaults to ``False``.
        metric_reg (float): Coefficient of metric regularization :footcite:`Kostic2023DPNets` :math:`\\mathcal{P}`
        use_relaxed_loss (bool): Whether to use the relaxed loss :footcite:`Kostic2023DPNets` :math:`\\mathcal{S}` in place of the projection loss. It is advisable to set ``use_relaxed_score = True`` in numerically challenging scenarios. Defaults to ``False``. 
        loss_coefficient (float): Coefficient premultiplying the loss function. Defaults to ``1.0``.
        metric_reg_type (str): Type of metric regularization. Can be either ``'fro'``, ``'von_neumann'`` or ``'log_fro'``. Defaults to ``'fro'``. :guilabel:`TODO - ADD REF`.
        optimizer_fn (torch.optim.Optimizer): Any Totch optimizer.
        optimizer_kwargs (dict): Hyperparameters used for initializing the optimizer.
        scheduler_fn (torch.optim.lr_scheduler.LRScheduler): A Torch learning rate scheduler function.
        scheduler_kwargs (dict): Hyperparameters used for initializing the scheduler.
        scheduler_config: Configuration of the scheduler. Please refer to the `Lightning documentation <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_ for additional
            informations on how to configure the scheduler configuration (``lr_scheduler_config``).
        
    """
    def __init__(
            self,
            encoder: Type[nn.Module], encoder_kwargs: dict,
            weight_sharing: bool,
            metric_reg: float,
            use_relaxed_loss: bool,
            loss_coefficient: float,
            metric_reg_type: str,
            optimizer_fn: Type[torch.optim.Optimizer], optimizer_kwargs: dict,
            scheduler_fn: Type[torch.optim.lr_scheduler.LRScheduler], scheduler_kwargs: dict,
            scheduler_config: dict,
    ):
        super().__init__()
        for k, v in encoder_kwargs.items():
            self.hparams[f'encoder_{k}'] = v
        for k, v in optimizer_kwargs.items():
            self.hparams[f'optim_{k}'] = v
        for k, v in scheduler_kwargs.items():
            self.hparams[f'sched_{k}'] = v
        for k, v in scheduler_config.items():
            self.hparams[f'sched_{k}'] = v
        self.save_hyperparameters()
        
        self.weigth_sharing = weight_sharing
        self.encoder_init = encoder(**encoder_kwargs)        
        # If weight_sharing is True, the input and output encoders are the same (with shared weights).
        if not self.weight_sharing:
            self.encoder_evolved = encoder(**encoder_kwargs)
        else:
            self.encoder_evolved = self.encoder_init
        
        self.metric_reg = metric_reg
        self.use_relaxed_loss = use_relaxed_loss
        self.loss_coefficient = loss_coefficient
        self.metric_reg_type = metric_reg_type

        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_fn is not None:
            scheduler = self.scheduler_fn(optimizer, **self.scheduler_kwargs)
            lr_scheduler_config = self.scheduler_config.copy()
            lr_scheduler_config['scheduler'] = scheduler
            return [optimizer], [lr_scheduler_config]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self.base_step(train_batch)

    def validation_step(self, valid_batch, batch_idx):
        return self.base_step(valid_batch)

    def forward(self, X: Type[torch.Tensor]):
        return self.encoder_init(X)

    def base_step(self, batch, with_metrics: bool = True):
        """Default step (train loop) used for training and validation."""    
        X, Y = batch
        encoded_X = self.encoder_init(X)
        # encoder_evolved = encoder_init if self.weigth_sharing == True.
        encoded_Y = self.encoder_evolved(Y)

        _norm = torch.rsqrt(torch.tensor(X.shape[0]))
        encoded_X *= _norm
        encoded_Y *= _norm

        cov_X = torch.mm(encoded_X.T, encoded_X)
        cov_Y = torch.mm(encoded_Y.T, encoded_Y)
        cov_XY = torch.mm(encoded_X.T, encoded_Y)

        if self.metric_reg_type == 'fro':
            reg_fn = fro_reg
        elif self.metric_reg_type == 'von_neumann':
            reg_fn = von_neumann_reg
        elif self.metric_reg_type == 'log_fro':
            reg_fn = log_fro_reg
        else:
            raise ValueError('metric_reg_type must be either fro, von_neumann or log_fro')

        loss = self.metric_reg*reg_fn(cov_X, cov_Y)
        if self.use_relaxed_loss:
            loss += self.loss_coefficient*relaxed_projection_loss(cov_X, cov_Y, cov_XY)
        else:
            loss += self.loss_coefficient*projection_loss(cov_X, cov_Y, cov_XY)
        
        outputs = {
            'loss': loss,
        }
        #Create metrics dict setting no_grad=True to avoid memory leaks
        if with_metrics:
            with torch.no_grad():
                metrics = {
                    'projection_score': projection_loss(cov_X, cov_Y, cov_XY),
                    'relaxed_projection_score': relaxed_projection_loss(cov_X, cov_Y, cov_XY),
                    'reg': reg_fn(cov_X, cov_Y),
                    'condition_number': torch.linalg.cond(cov_X),
                    'rank': torch.linalg.matrix_rank(cov_X),
                }
            outputs['metrics'] = metrics
        return outputs
