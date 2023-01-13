#Training boilerplate

#Imports

#Standard Library
from typing import NamedTuple
from numbers import Number
import json

#JAX
import jax.scipy
import jax.numpy as jnp
import haiku as hk
import optax

#MISC
from einops import einsum
import sympy.abc
from sympy import fps, ln

#Functions
class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState

class ReducedRankGEPResults(NamedTuple):
    svals_sq: jnp.ndarray
    revecs: jnp.ndarray

with open('config.json') as f:
    CONFIG = json.load(f)

#Defining the feature map architecture
def _net(x):
    net = hk.nets.MLP(output_sizes=CONFIG['network']['layer_widths'], name=CONFIG['network']['name'])
    return net(x)
feature_map = hk.without_apply_rng(hk.transform(_net))

#Defining the optimizer
optimiser = optax.adamw(CONFIG['opt']['learning_rate'])

#Defining the loss function
def time_lagged_covariance(params: hk.Params, data: jnp.ndarray) -> jnp.ndarray:
    """Generate time-lagged covariances

    Args:
        params (hk.Params): haiku params to be injected into the feature map
        data (jnp.ndarray): array of shape [num_lagtimes, batch_size, num_in_features]

    Returns:
        jnp.ndarray: array of shape [num_lagtimes, num_out_features, num_out_features] storing time-lagged covariance matrices.
    """
    featurized_data = feature_map.apply(params=params, x=data)
    return einsum(featurized_data[:, 0, :], featurized_data, 'pts feats, pts lags feats_lagged -> lags feats feats_lagged')

#Kernel Ridge Regression of log(T + Id), T being the Transfer operator.
def reduced_rank_GEP(params: hk.Params, data: jnp.ndarray, series_coeffs: jnp.ndarray, reg: Number) -> ReducedRankGEPResults:
    cov_data = time_lagged_covariance(params, data)
    cov = cov_data[0] + reg*jnp.eye(cov_data.shape[1], dtype=cov_data.dtype)
    cross_cov = einsum(cov_data, series_coeffs, "i feats feats_lagged, i -> feats feats_lagged")
    svals_sq, rvecs = jax.scipy.linalg.eigh(cross_cov@(cross_cov.T), cov)
    sort_perm = jnp.flip(jnp.argsort(svals_sq))
    svals_sq = svals_sq[sort_perm]
    rvecs = rvecs[:, sort_perm]
    return ReducedRankGEPResults(svals_sq, rvecs) # svals_sq (+ rvecs) monotonically decreasing, i.e. trivial reduced_rank estimation.

def ln_series_coefficients(order: int) -> jnp.array:
    return jnp.flip(jnp.asarray(fps(ln(1 + sympy.abc.z), x0=0).polynomial(n=order + 1).as_poly().all_coeffs(), dtype=float))

def loss(params: hk.Params, data: jnp.ndarray, reg: Number, p: int = 2, rank: int = 0):
    order = data.shape[1] - 1 
    series_coeffs = ln_series_coefficients(order)
    svals = reduced_rank_GEP(params, data, series_coeffs, reg).svals_sq
    svals_p = jax.lax.pow(svals, 0.5*p)
    if rank <= 0:
        return -jnp.sum(svals_p)
    else:
        return -jnp.sum(svals_p[:rank])

#Training utils
@jax.jit
def update(state: TrainingState, data_batch: jnp.ndarray) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(state.params, data_batch, CONFIG['opt']['tikhonov_reg'], CONFIG['opt']['VAMP_order'], CONFIG['opt']['rank'])
    updates, opt_state = optimiser.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    avg_params = optax.incremental_update(params, state.avg_params, step_size=0.001)
    return TrainingState(params, avg_params, opt_state)