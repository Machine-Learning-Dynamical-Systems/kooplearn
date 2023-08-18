import warnings
from functools import partial
from typing import Optional, Union, Callable, Type

import numpy as np
import torch
from sklearn.utils import check_array, check_X_y
import lightning as L
from lightning.pytorch.loggers.logger import Logger

from kooplearn._src.deep_learning.lightning_modules.BruntonModule import BruntonModule
from kooplearn._src.deep_learning.loss_fns.brunton_loss import brunton_loss
from kooplearn._src.deep_learning.utils.Brunton_utils import AuxiliaryNetworkWrapper
from kooplearn._src.models.abc import BaseModel
from numpy.typing import ArrayLike


class BruntonModel(BaseModel):
    """Brunton Model for the Koopman operator.

    This model is based on [1] and uses a neural network to learn the Koopman operator. The neural network is composed
    of an autoencoder (encoder and decoder) and an auxiliary network. The encoder identify a few intrinsic coordinates
    y = φ(x) where the dynamics evolve, and the decoder recover the state x = φ−1(y). The auxiliary network is used to
    parametrize the matrix K that characterizes the dynamic system y[k+1] = K*y[k]. For more details, see [1].

    The model is implemented using PyTorch Lightning.

    [1] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. “Deep Learning for Universal Linear Embeddings of
    Nonlinear Dynamics.” Nature Communications 9, no. 1 (November 23, 2018): 4950.
    https://doi.org/10.1038/s41467-018-07210-0.

    Parameters:

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
        m_time_steps_linear_dynamics: Number of time steps m to enforce linear prediction, used in the linear dynamics
            loss term.
        m_time_steps_future_state_prediction: Number of time steps m to perform future state prediction, used in the
            future state prediction loss term.
        alpha_1: Weight of the reconstruction and future state prediction loss terms. Same notation as in [1].
        alpha_2: Weight of the infinity loss term. Same notation as in [1].
        alpha_3: Weight decay of the optimizer. Same notation as in [1].
        num_complex_pairs: Number of complex pairs eigenvalues parametrized by the auxiliary network.
        num_real: Number of real eigenvalues parametrized by the auxiliary network. Note that the num_complex_pairs +
            num_real must be equal to the dimension of the autoencoder subspace.
        optimizer_fn: Optimizer function. Can be any torch.optim.Optimizer.
        optimizer_hyperparameters: Hyperparameters of the optimizer. Must be a dictionary containing as keys the names
            of the hyperparameters and as values the values of the hyperparameters of the optimizer.
        scheduler_fn: Scheduler function. Can be any torch.optim.lr_scheduler.LRScheduler.
        scheduler_hyperparameters: Hyperparameters of the scheduler. Must be a dictionary containing as keys the names
            of the hyperparameters and as values the values of the hyperparameters of the scheduler.
        scheduler_config: Configuration of the scheduler. Must be a dictionary containing as keys the names of
            the configuration parameters and as values the values of the configuration parameters of the scheduler.
            See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers for more
            information on how to configure the scheduler configuration (lr_scheduler_config in their documentation).
        callbacks_fns: List of callback functions. Can be any lightning callback.
        callbacks_hyperparameters: List of dictionaries containing the hyperparameters of the callbacks. Must be a list
            of dictionaries containing as keys the names of the hyperparameters and as values the values of the
            hyperparameters of the callbacks in the order used in callbacks_fns.
        logger_fn: Logger function. Can be any lightning logger.
        logger_kwargs: Hyperparameters of the logger. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the logger.
        trainer_kwargs: Hyperparameters of the trainer. Must be a dictionary containing as keys the names of
            the hyperparameters and as values the values of the hyperparameters of the lightning trainer.
        seed: Seed for reproducibility.

    Attributes:
        dnn_model_module: PyTorch Lightning module containing the encoder, decoder and auxiliary network. The training
            loop is implemented in this module.
        datamodule: PyTorch Lightning datamodule containing the train, validation and test datasets.
        logger: PyTorch Lightning logger.
        trainer: PyTorch Lightning trainer.
        callbacks: List of PyTorch Lightning callbacks.
    """
    def __init__(
            self,
            num_complex_pairs: int,
            num_real: int,
            encoder_class: Type[torch.nn.Module],
            encoder_hyperparameters: dict,
            decoder_class: Type[torch.nn.Module],
            decoder_hyperparameters: dict,
            auxiliary_network_class: Type[torch.nn.Module],
            auxiliary_network_hyperparameters: dict,
            m_time_steps_linear_dynamics: int,
            m_time_steps_future_state_prediction: int,
            alpha_1: float,
            alpha_2: float,
            alpha_3: float,
            trainer_kwargs: dict,
            seed: int,
            optimizer_fn: Type[torch.optim.Optimizer], optimizer_hyperparameters: dict,
            scheduler_fn: Type[torch.optim.lr_scheduler.LRScheduler] = None, scheduler_hyperparameters: dict = None,
            scheduler_config: dict = None,
            callbacks_fns: list[Type[L.Callback]] = None, callbacks_hyperparameters: list[dict] = None,
            logger_fn: Type[Logger] = None, logger_kwargs: dict = None,
    ):
        self.encoder_class = encoder_class
        self.encoder_hyperparameters = encoder_hyperparameters
        self.decoder_class = decoder_class
        self.decoder_hyperparameters = decoder_hyperparameters
        self.auxiliary_network_class = auxiliary_network_class
        self.auxiliary_network_hyperparameters = auxiliary_network_hyperparameters
        self.auxiliary_network_wrapper = AuxiliaryNetworkWrapper
        self.auxiliary_network_wrapper_hyperparameters = {
            'model_architecture': auxiliary_network_class,
            'model_hyperparameters': auxiliary_network_hyperparameters,
            'num_complex_pairs': num_complex_pairs,
            'num_real': num_real,
        }
        self.m_time_steps_linear_dynamics = m_time_steps_linear_dynamics
        self.m_time_steps_future_state_prediction = m_time_steps_future_state_prediction
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
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
        self.dnn_model_module = None
        self.datamodule = None
        self.logger = None
        self.trainer = None
        self.callbacks = None

    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
        """Performs pre-fit checks on the training data.

        Use check_array and check_X_y from sklearn to check the training data, initialize the covariance matrices and
        save the training data.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.

        """
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)
        self.X_fit_ = X
        self.Y_fit_ = Y

    def initialize(self):
        """Initializes the logger, the lightning module, the callbacks and the trainer."""
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X: ArrayLike, Y: ArrayLike, datamodule: L.LightningDataModule = None):
        """Fits the Brunton model.

        A datamodule is required for this model.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.
            datamodule: Pytorch lightning datamodule.
        """
        self.pre_fit_checks(X, Y)
        # X = self.X_fit_
        # Y = self.Y_fit_
        self.datamodule = datamodule
        self.initialize()
        self.trainer.fit(model=self.dnn_model_module, datamodule=self.datamodule)

    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None) \
            -> np.ndarray:
        """Predicts the state at time = t + 1 given the current state X.

        Optionally can predict an observable of the state at time = t + 1.

        Parameters:
            X: Current state of the system, shape (n_samples, n_features).
            t: Number of steps to predict (return the last one).
            observables: TODO add description and check if we can use it.

        Returns:
            The predicted state at time = t + 1, shape (n_samples, n_features).

        """
        is_reshaped = False
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if X.shape[-1] == self.datamodule.lb_window_size * self.datamodule.train_dataset.values.shape[-1]:
            # In this case X is (n_samples, n_features*lb_window_size), but we want
            # (n_samples, n_features, lb_window_size)
            X = X.reshape(X.shape[0], -1, self.datamodule.lb_window_size)
            is_reshaped = True
        data = {'x_value': X}
        self.dnn_model_module.eval()
        with torch.no_grad():
            model_output = self.dnn_model_module(data)
        if is_reshaped:
            return model_output.reshape(X.shape[0], -1).detach().numpy()
        return model_output.detach().numpy()

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        """Computes the eigenvalues of the Koopman operator and optionally evaluate left eigenfunctions.

        Note that the Brunton model does not support evaluation of right eigenfunctions. As the eigenvalues are
        dependent of the state, if eval_left_on is None, the eigenvalues of the last training sample are returned.

        Parameters:
            eval_left_on: State of the system to evaluate the left eigenfunction on, shape (n_samples, n_features).
            eval_right_on: State of the system to evaluate the right eigenfunction on, shape (n_samples, n_features).

        Returns:
            Eigenvalues of the Koopman operator, shape (p,).
            Left eigenfunction evaluated at eval_left_on, shape (n_samples, p) if eval_left_on is not None.
            TODO check if shapes are correct.

        """
        if eval_right_on is not None:
            raise ValueError("BruntonModel does not support evaluation of right eigenfunctions.")
        if eval_left_on is None:
            warnings.warn("eval_left_on is None. Returning eigenvalues of last training sample.")
            X = self.X_fit_[-1]
        else:
            X = eval_left_on
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if X.shape[-1] == self.datamodule.lb_window_size * self.datamodule.train_dataset.values.shape[-1]:
            # In this case X is (n_samples, n_features*lb_window_size), but we want
            # (n_samples, n_features, lb_window_size)
            X = X.reshape(X.shape[0], -1, self.datamodule.lb_window_size)
        data = {'x_value': X}
        self.dnn_model_module.eval()
        with torch.no_grad():
            y = self.dnn_model_module.encoder(data)
            mus_omegas, lambdas = self.dnn_model_module.auxiliary_network(y)
            eigenvalues = torch.complex(mus_omegas[:, 0], mus_omegas[:, 1])
            eigenvalues = torch.cat([eigenvalues, lambdas], dim=-1)
        eigenvalues = eigenvalues.detach().numpy()
        y = y.detach().numpy()
        if eval_left_on is None:
            return eigenvalues
        return eigenvalues, y

    def modes(self, Xin: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        """Computes the modes of the system at the state X.

        Following the notation of [1] the modes are given by the block diagonal matrix K.
        Optionally can compute the modes of an observable of the system at the state X.
        TODO verify how and if it is correct

        Parameters:
            Xin: State of the system, shape (n_samples, n_features).
            observables: TODO add description if we can use them

        Returns:
            Modes of the system at the state X, shape (n_samples, p, p).
        """
        if observables is not None:
            raise ValueError("BruntonModel does not support evaluation of observables.")  # or check how to do it
        X = Xin
        if X.shape[-1] == self.datamodule.lb_window_size * self.datamodule.train_dataset.values.shape[-1]:
            # In this case X is (n_samples, n_features*lb_window_size), but we want
            # (n_samples, n_features, lb_window_size)
            X = X.reshape((X.shape[0], -1, self.datamodule.lb_window_size))
        data = {'x_value': X}
        self.dnn_model_module.eval()
        with torch.no_grad():
            y = self.dnn_model_module.encoder(data)
            mus_omegas, lambdas = self.dnn_model_module.auxiliary_network(y)
        cos_omega = torch.cos(mus_omegas[..., 1])
        sin_omega = torch.sin(mus_omegas[..., 1])
        jordan_block_complex = torch.stack([torch.cat([cos_omega, -sin_omega], dim=-1),
                                            torch.cat([sin_omega, cos_omega], dim=-1)],
                                           dim=-1)  # should be of shape (..., num_complex_pairs, 2, 2)
        jordan_block_complex = torch.exp(mus_omegas[..., 0]) * jordan_block_complex
        jordan_block_real = torch.exp(lambdas)
        modes = list(jordan_block_complex.unbind(dim=-3)) + list(jordan_block_real.unbind(dim=-1))
        modes = torch.block_diag(modes)
        return modes.detach().numpy()

    def initialize_logger(self):
        """Initializes the logger."""
        if self.logger_fn:
            self.logger = self.logger_fn(**self.logger_kwargs)
        else:
            self.logger = None

    def initialize_model_module(self):
        """Initializes the Brunton lightning module."""
        self.dnn_model_module = BruntonModule(
            encoder_class=self.encoder_class,
            encoder_hyperparameters=self.encoder_hyperparameters,
            decoder_class=self.decoder_class,
            decoder_hyperparameters=self.decoder_hyperparameters,
            auxiliary_network_class=self.auxiliary_network_wrapper,
            auxiliary_network_hyperparameters=self.auxiliary_network_wrapper_hyperparameters,
            m_time_steps_linear_dynamics=self.m_time_steps_linear_dynamics,
            m_time_steps_future_state_prediction=self.m_time_steps_future_state_prediction,
            optimizer_fn=self.optimizer_fn,
            optimizer_hyperparameters=self.optimizer_hyperparameters,
            scheduler_fn=self.scheduler_fn,
            scheduler_hyperparameters=self.scheduler_hyperparameters,
            scheduler_config=self.scheduler_config,
            loss_fn=partial(brunton_loss, alpha_1=self.alpha_1, alpha_2=self.alpha_2),
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
