import warnings
from functools import partial
from typing import Optional, Union, Callable

import numpy as np
import torch
from sklearn.utils import check_array, check_X_y
import lightning as L

from kooplearn._src.deep_learning.lightning_modules.BruntonModule import BruntonModule
from kooplearn._src.deep_learning.loss_fns.brunton_loss import brunton_loss
from kooplearn._src.models.abc import BaseModel
from numpy.typing import ArrayLike


class BruntonModel(BaseModel):
    def __init__(
            self,
            encoder_class,
            encoder_hyperparameters,
            decoder_class,
            decoder_hyperparameters,
            auxiliary_network_class,
            auxiliary_network_hyperparameters,
            m_time_steps_linear_dynamics,
            m_time_steps_future_state_prediction,
            alpha_1,
            alpha_2,
            alpha_3,
            num_complex_pairs,
            num_real,
            optimizer_fn, optimizer_kwargs,
            scheduler_fn, scheduler_kwargs, scheduler_config,
            callbacks_fns, callbacks_kwargs,
            logger_fn, logger_kwargs,
            trainer_kwargs,
            seed,
    ):
        self.encoder_class = encoder_class
        self.encoder_hyperparameters = encoder_hyperparameters
        self.decoder_class = decoder_class
        self.decoder_hyperparameters = decoder_hyperparameters
        self.auxiliary_network_class = auxiliary_network_class
        self.auxiliary_network_hyperparameters = auxiliary_network_hyperparameters
        self.m_time_steps_linear_dynamics = m_time_steps_linear_dynamics
        self.m_time_steps_future_state_prediction = m_time_steps_future_state_prediction
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config
        self.callbacks_fns = callbacks_fns
        self.callbacks_kwargs = callbacks_kwargs
        self.logger_fn = logger_fn
        self.logger_kwargs = logger_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.seed = seed
        self.dnn_model_module = None
        self.datamodule = None
        self.logger = None
        self.trainer = None
        self.callbacks = None

    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)
        self.X_fit_ = X
        self.Y_fit_ = Y

    def initialize(self):
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X: ArrayLike, Y: ArrayLike, datamodule=None):
        self.pre_fit_checks(X, Y)
        # X = self.X_fit_
        # Y = self.Y_fit_
        self.datamodule = datamodule
        self.initialize()
        self.trainer.fit(model=self.dnn_model_module, datamodule=self.datamodule)

    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
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
            return model_output['x_encoded'].reshape(X.shape[0], -1).detach().numpy()
        return model_output['x_encoded'].detach().numpy()

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
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
        self.logger = None

    def initialize_model_module(self):
        self.dnn_model_module = BruntonModule(
            encoder_class=self.encoder_class,
            encoder_hyperparameters=self.encoder_hyperparameters,
            decoder_class=self.decoder_class,
            decoder_hyperparameters=self.decoder_hyperparameters,
            auxiliary_network_class=self.auxiliary_network_class,
            auxiliary_network_hyperparameters=self.auxiliary_network_hyperparameters,
            m_time_steps_linear_dynamics=self.m_time_steps_linear_dynamics,
            m_time_steps_future_state_prediction=self.m_time_steps_future_state_prediction,
            optimizer_fn=self.optimizer_fn,
            optimizer_hyperparameters=self.optimizer_kwargs,
            scheduler_fn=self.scheduler_fn,
            scheduler_hyperparameters=self.scheduler_kwargs,
            scheduler_config=self.scheduler_config,
            loss_fn=partial(brunton_loss, alpha_1=self.alpha_1, alpha_2=self.alpha_2),
        )

    def initialize_callbacks(self):
        self.callbacks = [fn(**kwargs) for fn, kwargs in zip(self.callbacks_fns, self.callbacks_kwargs)]

    def initialize_trainer(self):
        self.trainer = L.Trainer(**self.trainer_kwargs, callbacks=self.callbacks, logger=self.logger)
