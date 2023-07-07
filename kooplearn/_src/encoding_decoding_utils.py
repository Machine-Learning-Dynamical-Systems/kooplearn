import abc
import torch
from typing import Optional
from numpy.typing import ArrayLike


class FeatureMap(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: ArrayLike):
        pass

    def cov(self, X: ArrayLike, Y: Optional[ArrayLike] = None):
        phi_X = self.__call__(X)
        if Y is None:
            c = phi_X.T @ phi_X
        else:
            phi_Y = self.__call__(Y)
            c = phi_X.T @ phi_Y
        c *= (X.shape[0]) ** (-1)
        return c


class IdentityFeatureMap(FeatureMap):
    def __call__(self, X: ArrayLike):
        return X


class TrainableFeatureMap(FeatureMap):
    # Trainable feature maps should accept numpy arrays and return numpy arrays. Internally thay can do whatever.
    @abc.abstractmethod
    def fit(self, X: Optional[ArrayLike], Y: Optional[ArrayLike]):
        pass


class LightningFeatureMap(TrainableFeatureMap):
    def __init__(self,
                 dnn_model_module_class,
                 dnn_model_class, dnn_model_kwargs,
                 optimizer_fn, optimizer_kwargs,
                 scheduler_fn, scheduler_kwargs, scheduler_config,
                 callbacks_fns, callbacks_kwargs,
                 logger_fn, logger_kwargs,
                 trainer_kwargs,
                 loss_fn,
                 seed,
                 ):
        super().__init__()
        self.dnn_model_module_class = dnn_model_module_class
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
        L.seed_everything(seed)
        self.logger = None
        self.datamodule = None
        self.dnn_model_module = None
        self.callbacks = None
        self.trainer = None
        self.feature_map = None
        self.koopman_estimator = None
        self.decoder = None
        self.dataset = None

    def initialize_logger(self):
        self.logger = self.logger_fn(**self.logger_kwargs)
        # log what is not logged by default using pytorch lightning
        self.logger.log_hyperparams({'seed': self.seed})
        self.logger.log_hyperparams(self.trainer_kwargs)
        for kwargs in self.callbacks_kwargs:
            self.logger.log_hyperparams(kwargs)

    def initialize_model_module(self):
        dnn_model_class = self.dnn_model_class
        self.datamodule.setup('fit')
        train_dataset = self.datamodule.train_dataset
        model_kwargs_from_dataset = dnn_model_class.time_series_dataset_to_model_kwargs(train_dataset)
        self.dnn_model_kwargs.update(model_kwargs_from_dataset)
        optimizer_kwargs_from_dataset = dnn_model_class.time_series_dataset_to_optimizer_kwargs(train_dataset)
        self.optimizer_kwargs.update(optimizer_kwargs_from_dataset)
        scheduler_kwargs_from_dataset = dnn_model_class.time_series_dataset_to_scheduler_kwargs(train_dataset)
        self.scheduler_kwargs.update(scheduler_kwargs_from_dataset)
        # koopman_estimator_kwargs_from_dataset = dnn_model_class.time_series_dataset_to_koopman_estimator_kwargs(
        #     train_dataset)
        # self.koopman_estimator_kwargs.update(koopman_estimator_kwargs_from_dataset)
        self.dnn_model_module = self.dnn_model_module_class(
            model_class=dnn_model_class,
            model_hyperparameters=self.dnn_model_kwargs,
            dataset=train_dataset,
            optimizer_fn=self.optimizer_fn,
            optimizer_hyperparameters=self.optimizer_kwargs,
            koopman_estimator=self.koopman_estimator,
            decoder=self.decoder,
            # koopman_estimator_hyperparameters=self.koopman_estimator_kwargs,
            # decoder_hyperparameters=self.decoder_kwargs,
            scheduler_fn=self.scheduler_fn,
            scheduler_hyperparameters=self.scheduler_kwargs,
            scheduler_config=self.scheduler_config,
            loss_fn=self.loss_fn,
        )

    def initialize_callbacks(self):
        self.callbacks = [fn(**kwargs) for fn, kwargs in zip(self.callbacks_fns, self.callbacks_kwargs)]

    def initialize_trainer(self):
        self.trainer = L.Trainer(**self.trainer_kwargs, callbacks=self.callbacks, logger=self.logger)

    def initialize(self, koopman_estimator, decoder, datamodule):
        self.koopman_estimator = koopman_estimator
        self.decoder = decoder
        self.datamodule = datamodule
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X=None, Y=None):  # X and Y not used, here as placeholder to match the general case
        if self.datamodule is None:
            raise ValueError('Datamodule is required to use DNNFeatureMap.')
        self.trainer.fit(self.dnn_model_module, self.datamodule)

    def __call__(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        return self.dnn_model_module(X).detach().numpy()  # Everything should be outputted as a Numpy array


# !! Implement this
class Decoder(abc.ABC):
    """
    Decoder class, inverse operation of the feature map.

    TODO: implement this class in the case where feature map is a DPNet.
    """

    def fit_feature_map(self, X, Y):
        # may not need X or Y for setting up, here as placeholder in the general case
        pass

    def __call__(self, X):
        return X
