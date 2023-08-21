import numpy as np
import pandas as pd
from kooplearn._src.deep_learning.data_utils.TimeseriesDataModule import TimeseriesDataModule
from kooplearn._src.deep_learning.feature_maps.DPNetFeatureMap import DPNetFeatureMap
from kooplearn._src.deep_learning.architectures.MLPModel import MLPModel
from kooplearn._src.models.encoder_decoder import EncoderModel
import torch


def fit_dpnet_model_example(num_samples, num_features, lb_window_size, rank, epochs):
    seed = 42
    rng = np.random.default_rng(seed)
    data = rng.random((num_samples, num_features))
    df_series = pd.DataFrame(data).reset_index(names='time_idx')
    n_valid = int(0.2 * num_samples)
    n_test = int(0.2 * num_samples)
    n_train = num_samples - n_valid - n_test
    datamodule = TimeseriesDataModule(df_series=df_series, n_train=n_train, n_valid=n_valid, n_test=n_test,
                                      lb_window_size=lb_window_size, batch_size=32)
    encoder_input_class = MLPModel
    encoder_input_hyperparameters = {
        'input_dim': lb_window_size * num_features,
        'output_dim': 32,
        'hidden_dims': [32, 32],
        'flatten_input': True,
    }
    optimizer_fn = torch.optim.AdamW
    optimizer_hyperparameters = {}
    trainer_kwargs = {
        'max_epochs': epochs,
        'gradient_clip_val': 1.0,
    }
    seed = 42
    encoder_output_class = encoder_input_class
    encoder_output_hyperparameters = encoder_input_hyperparameters.copy()
    p_loss_coef = 1.0
    s_loss_coef = 1.0
    reg_1_coef = 1.0
    reg_2_coef = 1.0
    rank = rank
    feature_map = DPNetFeatureMap(
        encoder_input_class=encoder_input_class,
        encoder_input_hyperparameters=encoder_input_hyperparameters,
        encoder_output_class=encoder_output_class,
        encoder_output_hyperparameters=encoder_output_hyperparameters,
        optimizer_fn=optimizer_fn,
        optimizer_hyperparameters=optimizer_hyperparameters,
        trainer_kwargs=trainer_kwargs,
        seed=seed,
        p_loss_coef=p_loss_coef,
        s_loss_coef=s_loss_coef,
        reg_1_coef=reg_1_coef,
        reg_2_coef=reg_2_coef,
        rank=rank,
    )
    model = EncoderModel(feature_map=feature_map, rank=rank)
    datamodule.setup('fit')
    train_dataset = datamodule.train_dataset
    X, Y = train_dataset.get_numpy_matrices()
    model.fit(X=X, Y=Y, datamodule=datamodule)
    return model, X


def test_fit():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    epochs = 10
    rank = 6
    model, X = fit_dpnet_model_example(num_samples, num_features, lb_window_size, rank, epochs)
    assert model.feature_map.is_fitted
    assert model.U_ is not None


def test_predict():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    epochs = 2
    rank = 6
    model, X = fit_dpnet_model_example(num_samples, num_features, lb_window_size, rank, epochs)
    X_to_predict = X[0, :]
    prediction = model.predict(X_to_predict)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (num_features * lb_window_size,)
    prediction = model.predict(X_to_predict, t=2)
    assert prediction.shape == (num_features * lb_window_size,)
    X_to_predict = X[0:1, :]
    prediction = model.predict(X_to_predict)
    assert prediction.shape == (1, num_features * lb_window_size)
    X_to_predict = X[0:5, :]
    prediction = model.predict(X_to_predict)
    assert prediction.shape == (5, num_features * lb_window_size)


def test_eig():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    epochs = 2
    rank = 6
    model, X = fit_dpnet_model_example(num_samples, num_features, lb_window_size, rank, epochs)
    eig = model.eig()
    assert isinstance(eig, np.ndarray)
    assert eig.shape == (rank, )
    X_to_eval_left_on = X[0, :]  # shape (100,)
    eig = model.eig(eval_left_on=X_to_eval_left_on)
    assert isinstance(eig[0], np.ndarray)
    assert isinstance(eig[1], np.ndarray)
    assert eig[0].shape == (rank, )
    assert eig[1].shape == (rank, )
    X_to_eval_right_on = X[0:1, :]  # shape (1, 100)
    eig = model.eig(eval_right_on=X_to_eval_right_on)
    assert eig[0].shape == (rank,)
    assert eig[1].shape == (1, rank)
    X_to_eval_right_on = X[0:5, :]  # shape (5, 100)
    eig = model.eig(eval_right_on=X_to_eval_right_on)
    assert eig[0].shape == (rank, )
    assert eig[1].shape == (5, rank)
    X_to_eval_right_on = X[0, :]  # shape (100,)
    eig = model.eig(eval_right_on=X_to_eval_right_on)
    assert isinstance(eig[0], np.ndarray)
    assert isinstance(eig[1], np.ndarray)
    assert eig[0].shape == (rank, )
    assert eig[1].shape == (rank, )
    X_to_eval_right_on = X[0:1, :]  # shape (1, 100)
    eig = model.eig(eval_right_on=X_to_eval_right_on)
    assert eig[0].shape == (rank,)
    assert eig[1].shape == (1, rank)
    X_to_eval_right_on = X[0:5, :]  # shape (5, 100)
    eig = model.eig(eval_right_on=X_to_eval_right_on)
    assert eig[0].shape == (rank, )
    assert eig[1].shape == (5, rank)
    X_to_eval_right_on = X[0, :]  # shape (100,)
    X_to_eval_left_on = X[0:5, :]  # shape (5, 100)
    eig = model.eig(eval_right_on=X_to_eval_right_on, eval_left_on=X_to_eval_left_on)
    assert isinstance(eig[0], np.ndarray)
    assert isinstance(eig[1], np.ndarray)
    assert isinstance(eig[2], np.ndarray)
    assert eig[0].shape == (rank, )
    assert eig[1].shape == (5, rank)
    assert eig[2].shape == (rank, )
