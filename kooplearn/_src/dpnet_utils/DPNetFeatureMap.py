import lightning as L
import torch
from kooplearn._src.dpnet_utils.TimeseriesDataModule import TimeseriesDataModule
from .DPNetModule import DPNetModule

class DPNetFeatureMap:
    def __init__(self,
                 dnn_model_class, dnn_model_kwargs,
                 optimizer_fn, optimizer_kwargs,
                 scheduler_fn, scheduler_kwargs, scheduler_config,
                 callbacks_fns, callbacks_kwargs,
                 logger_fn, logger_kwargs,
                 trainer_kwargs,
                 seed,
                 loss_fn,
                 dnn_model_class_2=None, dnn_model_kwargs_2=None,
                 ):
        self.dnn_model_module_class = DPNetModule
        self.dnn_model_class = dnn_model_class
        self.dnn_model_kwargs = dnn_model_kwargs
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
        self.loss_fn = loss_fn
        self.seed = seed
        self.dnn_model_class_2 = dnn_model_class_2
        self.dnn_model_kwargs_2 = dnn_model_kwargs_2
        L.seed_everything(seed)
        self.logger = None
        self.datamodule = None
        self.dnn_model_module = None
        self.callbacks = None
        self.trainer = None
        self.model = None
        self.is_fitted = False

    def initialize_logger(self):
        # for the moment we will not use the logger
        pass
        # self.logger = self.logger_fn(**self.logger_kwargs)
        # # log what is not logged by default using pytorch lightning
        # self.logger.log_hyperparams({'seed': self.seed})
        # self.logger.log_hyperparams(self.trainer_kwargs)
        # for kwargs in self.callbacks_kwargs:
        #     self.logger.log_hyperparams(kwargs)

    def initialize_model_module(self):
        self.dnn_model_module = self.dnn_model_module_class(
            model_class=self.dnn_model_class, model_hyperparameters=self.dnn_model_kwargs,
            optimizer_fn=self.optimizer_fn, optimizer_hyperparameters=self.optimizer_kwargs, loss_fn=self.loss_fn,
            scheduler_fn=self.scheduler_fn, scheduler_hyperparameters=self.scheduler_kwargs,
            scheduler_config=self.scheduler_config,
            model_class_2=self.dnn_model_class_2, model_hyperparameters_2=self.dnn_model_kwargs_2,
        )

    def initialize_callbacks(self):
        self.callbacks = [fn(**kwargs) for fn, kwargs in zip(self.callbacks_fns, self.callbacks_kwargs)]

    def initialize_trainer(self):
        self.trainer = L.Trainer(**self.trainer_kwargs, callbacks=self.callbacks, logger=self.logger)

    def initialize(self):
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X=None, Y=None, datamodule=None):
        if datamodule is None:
            raise ValueError('Datamodule is required to use DNNFeatureMap.')
        self.datamodule = datamodule
        self.trainer.fit(model=self.dnn_model_module, datamodule=self.datamodule)
        self.is_fitted = True

    def __call__(self, X):
        is_reshaped = False
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if X.shape[-1] == self.datamodule.lb_window_size*self.datamodule.train_dataset.values.shape[-1]:
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
        return model_output['x_encoded'].detach().numpy()  # Everything should be outputted as a Numpy array
    # In the case where X is the entire dataset, we should implement a dataloader to avoid memory issues
    # (prediction on batches). For this we should implement a predict_step and call predict on the trainer.

    def cov(self, X, Y=None):
        phi_X = self.__call__(X)
        if Y is None:
            c = phi_X.T @ phi_X
        else:
            phi_Y = self.__call__(Y)
            c = phi_X.T @ phi_Y
        c *= (X.shape[0]) ** (-1)
        return c