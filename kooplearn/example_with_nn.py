# What I imagine would be the workflow for a user in the case of a DNN

from kooplearn.main import GeneralModel
from kooplearn.Datasets.TimeseriesDataModule import TimeseriesDataModule
from kooplearn.feature_maps.DNNFeatureMap import DNNFeatureMap
from kooplearn.DNNs.torch.MLPModel import MLPModel
from kooplearn.DNNs.torch.KoopmanForecasterModule import KoopmanForecasterModule
from kooplearn.feature_maps.Decoder import Decoder
from kooplearn.koopman_estimators.DirectEstimators import DirectRegressor
import numpy as np
import pandas as pd

# Generate some sample data
n_samples = 1000
t = np.linspace(0, 10, n_samples)
x1 = np.sin(t)
x2 = np.cos(t)

# define datamodule
df = pd.DataFrame({'x1': x1, 'x2': x2})
n_valid = n_samples*0.1
n_test = n_samples*0.2
n_train = n_samples - n_valid - n_test
lb_window_size = 10
horizon_size = 5
batch_size = 32
datamodule = TimeseriesDataModule(df_series=df, n_train=n_train, n_valid=n_valid, n_test=n_test,
                                  lb_window_size=lb_window_size, horizon_size=horizon_size, batch_size=batch_size)

# define feature map
dnn_model_module_class = KoopmanForecasterModule
dnn_model_class = MLPModel
dnn_model_kwargs = {}  # dataset dependent hyperparameters will be initialized automatically

# other parameters can be initialized by default or customized by the user
optimizer_fn = dnn_model_class.get_default_optimizer_fn()
optimizer_kwargs = dnn_model_class.get_default_optimizer_kwargs()
scheduler_fn = dnn_model_class.get_default_scheduler_fn()
scheduler_kwargs = dnn_model_class.get_default_scheduler_kwargs()
scheduler_config = dnn_model_class.get_default_scheduler_config()
callbacks_fns = dnn_model_class.get_default_callbacks_fns()
callbacks_kwargs = dnn_model_class.get_default_callbacks_kwargs()
logger_fn = dnn_model_class.get_default_logger_fn()
logger_kwargs = dnn_model_class.get_default_logger_kwargs()
trainer_kwargs = dnn_model_class.get_default_trainer_kwargs()
loss_fn = dnn_model_class.get_default_loss_fn()
seed = 0

feature_map = DNNFeatureMap(dnn_model_module_class=dnn_model_module_class, dnn_model_class=dnn_model_class,
                            dnn_model_kwargs=dnn_model_kwargs, optimizer_fn=optimizer_fn, optimizer_kwargs=optimizer_kwargs,
                            scheduler_fn=scheduler_fn, scheduler_kwargs=scheduler_kwargs, scheduler_config=scheduler_config,
                            callbacks_fns=callbacks_fns, callbacks_kwargs=callbacks_kwargs, logger_fn=logger_fn,
                            logger_kwargs=logger_kwargs, trainer_kwargs=trainer_kwargs, loss_fn=loss_fn, seed=seed)

# define kooopman estimator
koopman_estimator_hyperparameters = dict()
koopman_estimator = DirectRegressor(**koopman_estimator_hyperparameters)

# define decoder
decoder_hyperparameters = dict()
decoder = Decoder(**decoder_hyperparameters)

# define general model
model = GeneralModel(feature_map=feature_map, koopman_estimator=koopman_estimator, decoder=decoder)

# fit model
model.fit(datamodule=datamodule)

# In this case we imagine that we are fitting the feature map when fitting the model, so behind the scenes, the feature
# map is being fitted by
# GeneralModel.fit_feature_map() -> DNNFeatureMap.fit() and the training loop is defined by
# KoopmanForecasterModule.training_step() which calls KoopmanForecasterModule.base_step(),
# So if we want to inject information from the forecast task into the feature map, we have to modify the
# base_step() method


# test model
model.test(datamodule=datamodule)
