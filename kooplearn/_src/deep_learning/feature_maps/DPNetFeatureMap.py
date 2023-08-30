from functools import partial
from typing import Type
import lightning as L
from lightning.pytorch.loggers.logger import Logger
import torch
from torch import nn
from kooplearn._src.deep_learning.lightning_modules.DPNetModule import DPNetModule
from kooplearn._src.deep_learning.loss_fns.dpnet_loss import dpnets_loss
from kooplearn.abc import TrainableFeatureMap



class DPNetFeatureMap(TrainableFeatureMap):
    """Feature map to be used with EncoderModel to create a DPNet model.

    Trainable feature map based on [1]. The feature map is based on two neural networks, one for encoding the input
    and another for encoding the output. In case we do not specify the encoder for the output, we use the same model
    as the encoder for the input (with shared weights).

    The feature map is implemented using Pytorch Lightning.

    [1] Vladimir Kostic, Pietro Novelli, Riccardo Grazzi, Karim Lounici, and Massimiliano Pontil. “Deep
    Projection Networks for Learning Time-Homogeneous Dynamical Systems.” arXiv, July 19,
    2023. https://doi.org/10.48550/arXiv.2307.09912.

    Parameters:
        encoder_input_class: Class of the neural network used for encoding the input. Can be any deep learning
            architecture (torch.nn.Module) that takes as input a dictionary containing the key 'x_value', a tensor of
            shape (..., n_features, temporal_dim), and encodes it into a tensor of shape (..., output_dimension).
        encoder_input_hyperparameters: Hyperparameters of the neural network used for encoding the input. Must be a dictionary
            containing as keys the names of the hyperparameters and as values the values of the hyperparameters of the
            encoder.
        optimizer_fn: Optimizer function. Can be any torch.optim.Optimizer.
        optimizer_hyperparameters: Hyperparameters of the optimizer. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the optimizer.
        scheduler_fn: Scheduler function. Can be any torch.optim.lr_scheduler.LRScheduler.
        scheduler_hyperparameters: Hyperparameters of the scheduler. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the scheduler.
        scheduler_config: Configuration of the scheduler. Must be a dictionary containing as keys the names of
            the configuration parameters and as values the values of the configuration parameters of the scheduler.
            See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers for more
            information on how to configure the scheduler configuration (lr_scheduler_config in their documentation).
        callbacks_fns: List of callback functions. Can be any lightning callback.
        callbacks_hyperparameters: List of dictionaries containing the hyperparameters of the callbacks. Must be a list of
            dictionaries containing as keys the names of the hyperparameters and as values the values of the
            hyperparameters of the callbacks in the order used in callbacks_fns.
        logger_fn: Logger function. Can be any lightning logger.
        logger_kwargs: Hyperparameters of the logger. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the logger.
        trainer_kwargs: Hyperparameters of the trainer. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the lightning trainer.
        seed: Seed for reproducibility.
        encoder_output_class: Class of the neural network used for encoding the output. Can be any deep learning
            architecture (torch.nn.Module) that takes as input a dictionary containing the key 'x_value', a tensor of
            shape (..., n_features, temporal_dim), and encodes it into a tensor of shape (..., output_dimension).
        encoder_output_hyperparameters: Hyperparameters of the neural network used for encoding the output. Must be a
            dictionary containing as keys the names of the hyperparameters and as values the values of the
            hyperparameters of the encoder.
        p_loss_coef: Coefficient of the score function P.
        s_loss_coef: Coefficient of the score function S.
        reg_1_coef: Coefficient of the regularization term 1.
        reg_2_coef: Coefficient of the regularization term 2.
        rank: Rank of the estimator (same as passed for the koopman estimator). Only needed in case we use the
            regularization term 2.
    """
    def __init__(
            self,
            encoder_input_class: Type[nn.Module], encoder_input_hyperparameters: dict,
            optimizer_fn: Type[torch.optim.Optimizer], optimizer_hyperparameters: dict,
            trainer_kwargs: dict,
            seed: int,
            scheduler_fn: Type[torch.optim.lr_scheduler.LRScheduler] = None, scheduler_hyperparameters: dict = None,
            scheduler_config: dict = None,
            callbacks_fns: list[Type[L.Callback]] = None, callbacks_hyperparameters: list[dict] = None,
            logger_fn: Type[Logger] = None, logger_kwargs: dict = None,
            encoder_output_class: Type[nn.Module] = None, encoder_output_hyperparameters: dict = None,
            p_loss_coef: float = 1.0,
            s_loss_coef: float = 0,
            reg_1_coef: float = 0,
            reg_2_coef: float = 0,
            rank: int = None,
    ):
        self.encoder_input_class = encoder_input_class
        self.encoder_input_hyperparameters = encoder_input_hyperparameters
        self.optimizer_fn = optimizer_fn
        self.optimizer_hyperparameters = optimizer_hyperparameters
        self.scheduler_fn = scheduler_fn
        self.scheduler_hyperparameters = scheduler_hyperparameters if scheduler_hyperparameters else {}
        self.scheduler_config = scheduler_config if scheduler_config else {}
        self.callbacks_fns = callbacks_fns if callbacks_fns else []
        self.callbacks_hyperparameters = callbacks_hyperparameters if callbacks_hyperparameters else []
        self.logger_fn = logger_fn
        self.logger_kwargs = logger_kwargs if logger_kwargs else {}
        self.trainer_kwargs = trainer_kwargs
        self.seed = seed
        self.encoder_output_class = encoder_output_class
        self.encoder_output_hyperparameters = encoder_output_hyperparameters if encoder_output_hyperparameters else {}
        self.p_loss_coef = p_loss_coef
        self.s_loss_coef = s_loss_coef
        self.reg_1_coef = reg_1_coef
        self.reg_2_coef = reg_2_coef
        self.rank = rank
        self.loss_fn = partial(dpnets_loss, rank=rank, p_loss_coef=p_loss_coef, s_loss_coef=s_loss_coef,
                               reg_1_coef=reg_1_coef, reg_2_coef=reg_2_coef)
        L.seed_everything(seed)
        self.logger = None
        self.datamodule = None
        self.dnn_model_module = None
        self.callbacks = None
        self.trainer = None
        self.model = None
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    def initialize_logger(self):
        """Initializes the logger."""
        if self.logger_fn:
            self.logger = self.logger_fn(**self.logger_kwargs)
        else:
            self.logger = None

    def initialize_model_module(self):
        """Initializes the DPNet lightning module."""
        self.dnn_model_module = DPNetModule(
            encoder_input_class=self.encoder_input_class,
            encoder_input_hyperparameters=self.encoder_input_hyperparameters,
            optimizer_fn=self.optimizer_fn, optimizer_hyperparameters=self.optimizer_hyperparameters,
            loss_fn=self.loss_fn,
            scheduler_fn=self.scheduler_fn, scheduler_hyperparameters=self.scheduler_hyperparameters,
            scheduler_config=self.scheduler_config,
            encoder_output_class=self.encoder_output_class,
            encoder_output_hyperparameters=self.encoder_output_hyperparameters,
        )

    def initialize_callbacks(self):
        """Initializes the callbacks."""
        if self.callbacks_fns:
            self.callbacks = [fn(**kwargs) for fn, kwargs in zip(self.callbacks_fns, self.callbacks_hyperparameters)]
        else:
            self.callbacks = []

    def initialize_trainer(self):
        """Initializes the trainer."""
        self.trainer = L.Trainer(**self.trainer_kwargs, callbacks=self.callbacks, logger=self.logger)

    def initialize(self):
        """Initializes the feature map."""
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X: np.ndarray, Y: np.ndarray, datamodule: L.LightningDataModule = None):
        """Fits the DPNet feature map.

        A datamodule is required for this model.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.
            datamodule: Pytorch lightning datamodule.
        """
        if datamodule is None:
            raise ValueError('Datamodule is required to use DNNFeatureMap.')
        self.datamodule = datamodule
        self.trainer.fit(model=self.dnn_model_module, datamodule=self.datamodule)
        self._is_fitted = True

    def __call__(self, X):
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
