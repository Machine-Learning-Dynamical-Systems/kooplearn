from functools import partial
from typing import Type, Optional
import lightning as L
from lightning.pytorch.loggers.logger import Logger
import numpy as np
import torch
from torch import nn
from kooplearn.models.feature_maps.DPNets.module import DPNetModule
from kooplearn.models.feature_maps.DPNets.loss_and_reg_fns import dpnets_loss
from kooplearn.abc import TrainableFeatureMap
from torch.utils.data import TensorDataset, DataLoader

class DPNetsFeatureMap(TrainableFeatureMap):
    """Feature map to be used in conjunction to `kooplearn.models.EncoderModel` to create a DPNet model.

    Trainable feature map based on :footcite:`Kostic2023DPNets`. The feature map is based on two neural networks, one for encoding the initial states and another for encoding the evolved states.

    The feature map is implemented using `Pytorch Lightning <https://lightning.ai/>`_.

    Parameters:
        encoder (torch.nn.Module): Class of the neural network used for encoding the input. Can be any ``torch.nn.Module`` taking as input a dictionary containing the key 'x_value', a tensor of
            shape (..., n_features, temporal_dim), and encodes it into a tensor of shape (..., output_dimension).
        encoder_kwargs (dict): Hyperparameters used for initializing the encoder.
        proj_loss (float): Coefficient of the projection score :footcite:`Kostic2023DPNets` :math:`\\mathcal{P}`.
        relaxed_loss (float): Coefficient of the relaxed score :footcite:`Kostic2023DPNets` :math:`\\mathcal{S}`.
        metric_reg (float): Coefficient of metric regularization :footcite:`Kostic2023DPNets` :math:`\\mathcal{P}`
        optimizer_fn (torch.optim.Optimizer): Any Totch optimizer.
        optimizer_kwargs (dict): Hyperparameters used for initializing the optimizer.
        trainer_kwargs (dict): Keyword arguments used to initialize the `Lightning trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        metric_reg_type (str): Type of metric regularization. Can be either ``'fro'``, ``'von_neumann'`` or ``'log_fro'``. Defaults to ``'fro'``. :guilabel:`TODO - ADD REF`.
        weight_sharing (bool): Whether to share the weights between the encoder of the initial data and the encoder of the evolved data. As reported in :footcite:`Kostic2023DPNets`, this can be safely set to ``True`` when the dynamical system is time-reversal invariant. Defaults to ``False``.
        scheduler_fn (torch.optim.lr_scheduler.LRScheduler): A Torch learning rate scheduler function.
        scheduler_kwargs (dict): Hyperparameters used for initializing the scheduler.
        scheduler_config: Configuration of the scheduler. Please refer to the `Lightning documentation <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_ for additional
            informations on how to configure the scheduler configuration (``lr_scheduler_config``).
        callbacks_fns (list[Callable]): List of callback functions. Can be any `Lightning callback <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#callback>`_.
        callbacks_kwargs (list[dict]): List of dictionaries containing the hyperparameters for initializing each of the callbacks.
        logger_fn: Logger function. Can be any `Lightning logger <https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging>`_.
        logger_kwargs (dict): Hyperparameters used to initialize the logger.
        seed (int or None): Random number generator seed for reproducibility. Set to `None` to use a random seed. Defaults to ``None``.
    """
    def __init__(
            self,
            #Model
            encoder: Type[nn.Module], encoder_kwargs: dict,
            proj_loss: float,
            relaxed_loss: float,
            metric_reg: float,
            # Optimization
            optimizer_fn: Type[torch.optim.Optimizer], optimizer_kwargs: dict,
            #Pytorch Lightning 
            trainer_kwargs: dict,
            # Keyword Arguments
            metric_reg_type: str = 'fro',
            weight_sharing: bool = False,
            scheduler_fn: Type[torch.optim.lr_scheduler.LRScheduler] = None, scheduler_kwargs: dict = None,
            scheduler_config: dict = None,
            callbacks_fns: list[Type[L.Callback]] = None, callbacks_kwargs: list[dict] = None,
            logger_fn: Type[Logger] = None, logger_kwargs: dict = None,
            seed: Optional[int] = None
    ):
        #Set rng seed
        L.seed_everything(seed)
        self.seed = seed
        #Save parameters
        self.encoder = encoder
        self.encoder_kwargs = encoder_kwargs
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.scheduler_config = scheduler_config if scheduler_config else {}
        
        self.weight_sharing = weight_sharing
        self.proj_loss = proj_loss
        self.relaxed_loss = relaxed_loss
        self.metric_reg = metric_reg
        self.metric_reg_type = metric_reg_type

        #Init trainer
        self._initialize_callbacks(callbacks_fns, callbacks_kwargs)
        self._initialize_logger(logger_fn, logger_kwargs)
        self._initialize_trainer(trainer_kwargs)
        
        #Init model        
        
        
        #TODO
        self.loss_fn = partial(dpnets_loss, rank=rank, p_loss_coef=p_loss_coef, s_loss_coef=s_loss_coef,
                               reg_1_coef=reg_1_coef, reg_2_coef=reg_2_coef)
        
        self.dnn_model_module = None
        self.model = None
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    def initialize_model_module(self):
        """Initializes the DPNet lightning module."""
        self.dnn_model_module = DPNetModule(
            encoder=self.encoder,
            weight_sharing=self.weight_sharing,
            encoder_input_kwargs=self.encoder_input_kwargs,
            optimizer_fn=self.optimizer_fn, optimizer_kwargs=self.optimizer_kwargs,
            loss_fn=self.loss_fn,
            scheduler_fn=self.scheduler_fn, scheduler_kwargs=self.scheduler_kwargs,
            scheduler_config=self.scheduler_config,
        )

    def _initialize_logger(self, logger_fn, logger_kwargs):
        if logger_fn is not None:
            _kw = logger_kwargs if logger_kwargs else {}
            self.logger = logger_fn(**_kw)
        else:
            self.logger = None

    def _initialize_callbacks(self, callbacks_fns, callbacks_kwargs):
        """Initializes the callbacks."""
        if callbacks_fns is not None:
            if callbacks_kwargs is not None:
                assert len(callbacks_fns) == len(callbacks_kwargs)
            else:
                _kw = [{} for _ in callbacks_fns]
            self.callbacks = [fn(**kwargs) for fn, kwargs in zip(callbacks_fns, _kw)]
        else:
            self.callbacks = []

    def _initialize_trainer(self, trainer_kwargs):
        """Initializes the trainer."""
        self.trainer = L.Trainer(trainer_kwargs, callbacks=self.callbacks, logger=self.logger)

    def initialize(self):
        """Initializes the feature map."""
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None, datamodule: L.LightningDataModule = None):
        """Fits the DPNet feature map.

        A datamodule is required for this model.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.
            datamodule: Pytorch lightning datamodule.
        """
        if (X is not None) and (Y is not None):
            _X = torch.tensor(X, dtype=torch.float32)
            _Y = torch.tensor(Y, dtype=torch.float32)
            ds = TensorDataset(_X, _Y)
            #Create a full-batch dataloader from X and Y
            dl = DataLoader(ds, batch_size=len(ds))
            self.trainer.fit(model=self.dnn_model_module, train_dataloaders=dl)
        elif datamodule is not None:
            self.trainer.fit(model=self.dnn_model_module, datamodule=self.datamodule)
        else:
            raise ValueError('Either X and Y or datamodule must be specified.')
        self._is_fitted = True

    def __call__(self, X):
        #TODO use the "predict" method instead of this one
        """Applies the feature map to X."""
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if X.shape[-1] == self.datamodule.lb_window_size*self.datamodule.train_dataset.values.shape[-1]:
            # In this case X is (n_samples, n_features*lb_window_size), but we want
            # (n_samples, n_features, lb_window_size)
            X = X.unflatten(-1, (self.datamodule.train_dataset.values.shape[-1], self.datamodule.lb_window_size))
        self.dnn_model_module.eval()
        with torch.no_grad():
            model_output = self.dnn_model_module(X)
        return model_output.detach().numpy()  # Everything should be outputted as a Numpy array
    # In the case where X is the entire dataset, we should implement a dataloader to avoid memory issues
    # (prediction on batches). For this we should implement a predict_step and call predict on the trainer.
