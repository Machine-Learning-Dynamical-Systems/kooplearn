import numpy as np
import pandas as pd
from kooplearn._src.deep_learning.data_utils.TimeseriesDataModule import TimeseriesDataModule
from kooplearn._src.deep_learning.architectures.MLPModel import MLPModel
from kooplearn._src.deep_learning.models.brunton import BruntonModel
import torch


def fit_brunton_model_example(num_samples, num_features, lb_window_size, num_complex_pairs, num_real, epochs):
    seed = 42
    rng = np.random.default_rng(seed)
    data = rng.random((num_samples, num_features))
    df_series = pd.DataFrame(data).reset_index(names='time_idx')
    n_valid = int(0.2 * num_samples)
    n_test = int(0.2 * num_samples)
    n_train = num_samples - n_valid - n_test
    m_time_steps_linear_dynamics = 10
    m_time_steps_future_state_prediction = 5
    number_of_consecutive_time_steps_generated = max(m_time_steps_linear_dynamics, m_time_steps_future_state_prediction)
    datamodule = TimeseriesDataModule(df_series=df_series, n_train=n_train, n_valid=n_valid, n_test=n_test,
                                      lb_window_size=lb_window_size, batch_size=32,
                                      number_of_consecutive_time_steps_generated=number_of_consecutive_time_steps_generated)
    encoder_class = MLPModel
    encoder_hyperparameters = {
        'input_dim': lb_window_size * num_features,
        'output_dim': 2 * num_complex_pairs + num_real,
        'hidden_dims': [32, 32],
        'flatten_input': True,
    }
    decoder_class = encoder_class
    decoder_hyperparameters = {
        'input_dim': 2 * num_complex_pairs + num_real,
        'output_dim': lb_window_size * num_features,
        'hidden_dims': [32, 32],
        'flatten_input': False,
    }
    auxiliary_network_class = encoder_class
    auxiliary_network_hyperparameters = {
        'hidden_dims': [32, 32],
        'flatten_input': False,
    }
    alpha_1 = 1.0
    alpha_2 = 1.0
    alpha_3 = 1.0
    optimizer_fn = torch.optim.AdamW
    optimizer_hyperparameters = {
        'weight_decay': alpha_3,
    }
    trainer_kwargs = {
        'max_epochs': epochs,
    }
    seed = 42
    model = BruntonModel(
        num_complex_pairs=num_complex_pairs,
        num_real=num_real,
        encoder_class=encoder_class,
        encoder_hyperparameters=encoder_hyperparameters,
        decoder_class=decoder_class,
        decoder_hyperparameters=decoder_hyperparameters,
        auxiliary_network_class=auxiliary_network_class,
        auxiliary_network_hyperparameters=auxiliary_network_hyperparameters,
        m_time_steps_linear_dynamics=m_time_steps_linear_dynamics,
        m_time_steps_future_state_prediction=m_time_steps_future_state_prediction,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        alpha_3=alpha_3,
        optimizer_fn=optimizer_fn,
        optimizer_hyperparameters=optimizer_hyperparameters,
        trainer_kwargs=trainer_kwargs,
        seed=seed,
    )
    datamodule.setup('fit')
    train_dataset = datamodule.train_dataset
    X, Y = train_dataset.get_numpy_matrices()
    model.fit(X=X, Y=Y, datamodule=datamodule)
    return model, X


def test_fit():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    num_complex_pairs = 5
    num_real = 3
    model, X_train = fit_brunton_model_example(num_samples=num_samples, num_features=num_features, lb_window_size=lb_window_size,
                                      num_complex_pairs=num_complex_pairs, num_real=num_real, epochs=10)


def test_predict():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    num_complex_pairs = 5
    num_real = 3
    model, X_train = fit_brunton_model_example(num_samples=num_samples, num_features=num_features, lb_window_size=lb_window_size,
                                      num_complex_pairs=num_complex_pairs, num_real=num_real, epochs=2)
    X_to_predict = X_train[0, :]  # shape (100,)
    prediction = model.predict(X_to_predict)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (lb_window_size * num_features, )
    prediction = model.predict(X_to_predict, t=2)
    assert prediction.shape == (lb_window_size * num_features, )
    prediction = model.predict(X_to_predict, only_last_value=False)
    assert prediction.shape == (1, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, t=2, only_last_value=False)
    assert prediction.shape == (2, lb_window_size * num_features)
    X_to_predict = X_train[0:1, :]  # shape (1, 100)
    prediction = model.predict(X_to_predict)
    assert prediction.shape == (1, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, t=2)
    assert prediction.shape == (1, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, only_last_value=False)
    assert prediction.shape == (1, 1, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, t=2, only_last_value=False)
    assert prediction.shape == (1, 2, lb_window_size * num_features)
    X_to_predict = X_train[0:5, :]  # shape (5, 100)
    prediction = model.predict(X_to_predict)
    assert prediction.shape == (5, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, t=2)
    assert prediction.shape == (5, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, only_last_value=False)
    assert prediction.shape == (5, 1, lb_window_size * num_features)
    prediction = model.predict(X_to_predict, t=2, only_last_value=False)
    assert prediction.shape == (5, 2, lb_window_size * num_features)


def test_eig():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    num_complex_pairs = 5
    num_real = 3
    model, X_train = fit_brunton_model_example(num_samples=num_samples, num_features=num_features, lb_window_size=lb_window_size,
                                      num_complex_pairs=num_complex_pairs, num_real=num_real, epochs=2)
    X_to_eval_left_on = X_train[0, :]  # shape (100,)
    eig = model.eig(X_to_eval_left_on)
    assert isinstance(eig[0], np.ndarray)
    assert isinstance(eig[1], np.ndarray)
    assert eig[0].shape == (2*num_complex_pairs + num_real,)
    assert eig[1].shape == (2*num_complex_pairs + num_real,)
    eig = model.eig()
    assert eig.shape == (2*num_complex_pairs + num_real, )
    X_to_eval_left_on = X_train[0:1, :]  # shape (1, 100)
    eig = model.eig(X_to_eval_left_on)
    assert eig[0].shape == (1, 2*num_complex_pairs + num_real)
    assert eig[1].shape == (1, 2*num_complex_pairs + num_real)
    X_to_eval_left_on = X_train[0:5, :]  # shape (1, 100)
    eig = model.eig(X_to_eval_left_on)
    assert eig[0].shape == (5, 2*num_complex_pairs + num_real)
    assert eig[1].shape == (5, 2*num_complex_pairs + num_real)


def test_modes():
    num_samples = 1000
    num_features = 10
    lb_window_size = 10
    num_complex_pairs = 5
    num_real = 3
    model, X_train = fit_brunton_model_example(num_samples=num_samples, num_features=num_features,
                                               lb_window_size=lb_window_size,
                                               num_complex_pairs=num_complex_pairs, num_real=num_real, epochs=2)
    X_to_get_modes = X_train[0, :]  # shape (100,)
    modes = model.modes(X_to_get_modes)
    assert isinstance(modes, np.ndarray)
    assert modes.shape == (2*num_complex_pairs + num_real, 2*num_complex_pairs + num_real)
    X_to_get_modes = X_train[0:1, :]  # shape (1, 100)
    modes = model.modes(X_to_get_modes)
    assert modes.shape == (1, 2*num_complex_pairs + num_real, 2*num_complex_pairs + num_real)
    X_to_get_modes = X_train[0:5, :]  # shape (5, 100)
    modes = model.modes(X_to_get_modes)
    assert modes.shape == (5, 2*num_complex_pairs + num_real, 2*num_complex_pairs + num_real)
