# Authors: The kooplearn developers
from kooplearn.datasets._ordered_mnist import fetch_ordered_mnist
from kooplearn.datasets._samples_generator import (
    make_duffing,
    make_linear_system,
    make_logistic_map,
    make_lorenz63,
    make_prinz_potential,
    make_regime_switching_var,
)

__all__ = [
    "fetch_ordered_mnist",
    "make_duffing",
    "make_linear_system",
    "make_logistic_map",
    "make_lorenz63",
    "make_prinz_potential",
    "make_regime_switching_var",
]
