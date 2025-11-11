# Authors: The kooplearn developers
from kooplearn.datasets._logistic_map import (
    compute_logistic_map_eig,
    compute_logistic_map_invariant_pdf,
)
from kooplearn.datasets._ordered_mnist import fetch_ordered_mnist
from kooplearn.datasets._overdamped_langevin_generator import (
    compute_prinz_potential_eig,
)
from kooplearn.datasets._samples_generator import (
    make_duffing,
    make_linear_system,
    make_logistic_map,
    make_lorenz63,
    make_prinz_potential,
    make_regime_switching_var,
)

__all__ = [
    "compute_logistic_map_eig",
    "compute_logistic_map_invariant_pdf",
    "compute_prinz_potential_eig",
    "fetch_ordered_mnist",
    "make_duffing",
    "make_linear_system",
    "make_logistic_map",
    "make_lorenz63",
    "make_prinz_potential",
    "make_regime_switching_var",
]
