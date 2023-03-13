##Imports
from typing import Optional
import functools
#Scientific libs
import numpy as np
#JAX
import jax.scipy
import jax.scipy.optimize
import jax.numpy as jnp
import jax.lax
import jax

from jaxtyping import Float, Complex, Array

import optax
from torch.utils.tensorboard import SummaryWriter

#MISC
from einops import einsum 

from kooplearn.jax.typing import LinalgDecomposition, RealLinalgDecomposition
from kooplearn.jax.linalg import batch_spd_norm, generalized_eigh

def reduced_rank_regression(
    input_covariance: Float[Array, "d d"],
    cross_covariance: Float[Array, "d d"], #C_{XY}
    tikhonov_reg: float,
    ) -> RealLinalgDecomposition:
    """JAX-implementation of the reduced rank regression estimator.

    Args:
        input_covariance (Float[Array, &quot;d d&quot;]): Input Covariance matrix
        cross_covariance (Float[Array, &quot;d d&quot;]): Cross covariance matrix

    Returns:
        RealLinalgDecomposition: NamedTuple with the fitted estimator.
    """
    reg_input_covariance = input_covariance + tikhonov_reg*jnp.eye(input_covariance.shape[0], dtype=input_covariance.dtype)
    _crcov = cross_covariance@(cross_covariance.T) 
    _gep = generalized_eigh(_crcov, reg_input_covariance)
    
    _norms = batch_spd_norm(_gep.vectors, reg_input_covariance)
    vectors = _gep.vectors*(jax.lax.reciprocal(_norms))
    return RealLinalgDecomposition(_gep.values, vectors)

def randomized_reduced_rank_regression(
    input_covariance: Float[Array, "d d"],
    cross_covariance: Float[Array, "d d"], #C_{XY}
    tikhonov_reg: float,
    rank: int,
    key, #PRNGKey
    n_oversamples: int = 5, 
    iterated_power: int = 2,
    ) -> RealLinalgDecomposition:
    """JAX-implementation of the randomized reduced rank regression estimator.

    Args:
        input_covariance (Float[Array, &quot;d d&quot;]): Input Covariance Matrix
        cross_covariance (Float[Array, &quot;d d&quot;]): Cross covariance matrix
        rank (int): Rank of the estimator
        key (_type_): PRNG key
        iterated_power (int, optional): Number of power iterations to perform. Defaults to 2.

    Returns:
        RealLinalgDecomposition: NamedTuple with the fitted estimator.
    """    
    reg_input_covariance = input_covariance + tikhonov_reg*jnp.eye(input_covariance.shape[0], dtype=input_covariance.dtype)
    _crcov = cross_covariance@(cross_covariance.T)
    sketch = jax.random.normal(key, (reg_input_covariance.shape[0], rank + n_oversamples))

    for _ in range(iterated_power):
        _tmp_sketch = jax.scipy.linalg.solve(reg_input_covariance, sketch, assume_a='pos')
        sketch = _crcov@_tmp_sketch

    sketch_p =  jax.scipy.linalg.solve(reg_input_covariance, sketch, assume_a='pos')  
    F_0 = (sketch_p.T)@sketch
    F_1 = (sketch_p.T)@(_crcov@sketch_p)
    
    _gep = generalized_eigh(F_1, F_0)
    _norms = batch_spd_norm(_gep.vectors, F_0)
    vectors = _gep.vectors*(jax.lax.reciprocal(_norms))

    return RealLinalgDecomposition(_gep.values, sketch_p@vectors)

def tikhonov_regression(
    input_covariance: Float[Array, "d d"],
    tikhonov_reg: float,
    ) -> RealLinalgDecomposition:
    """JAX-implementation of the Tikhonov regression estimator.

    Args:
        input_covariance (Float[Array, &quot;d d&quot;]): Input Covariance matrix
        tikhonov_reg (float): Tikhonov regularization parameter

    Returns:
        RealLinalgDecomposition: NamedTuple with the fitted estimator.
    """    
    reg_input_covariance = input_covariance + tikhonov_reg*jnp.eye(input_covariance.shape[0], dtype=input_covariance.dtype)

    Lambda, Q = jnp.linalg.eigh(reg_input_covariance)
    rsqrt_Lambda = jnp.diag(jax.lax.rsqrt(Lambda))
    return RealLinalgDecomposition(Lambda, Q@rsqrt_Lambda)

def iterative_regression(
    input_data: Float[Array, "n d"],
    output_data: Float[Array, "n d"],
    num_iterations: int, 
    learning_rate: float=1.0, 
    momentum: bool = None, 
    nesterov: bool = False,
    tensorboard: bool = False
    ) -> Float[Array, "d d"]:    
    d = input_data.shape[1]
    estimator = jnp.zeros((d, d), dtype=input_data.dtype)

    optimizer = optax.sgd(learning_rate, momentum=momentum, nesterov=nesterov)
    opt_state = optimizer.init(estimator)

    @jax.jit
    def step(estimator, opt_state, input_batch, output_batch):
        loss_value, grads = jax.value_and_grad(sq_error, argnums = 2)(input_batch, output_batch, estimator)
        updates, opt_state = optimizer.update(grads, opt_state, estimator)
        estimator = optax.apply_updates(estimator, updates)
        return estimator, opt_state, loss_value
    if tensorboard:
        writer = SummaryWriter()
    for k in range(num_iterations):
        estimator, opt_state, loss_value = step(estimator, opt_state, input_data, output_data)
        if tensorboard and (k%10 == 0):
            writer.add_scalar("Loss/training", np.asarray(loss_value), k)
            _estim = np.asarray(estimator)
            _vals = np.linalg.eigvals(_estim)
            _leading_vals = np.sort_complex(_vals)[::-1][:3]
            writer.add_scalars("Eigenvalues/real_part", dict(zip(['1','2','3'], _leading_vals.real)), k)
            writer.add_scalars("Eigenvalues/imaginary_part", dict(zip(['1','2','3'], _leading_vals.imag)), k)
    return estimator

#Functions to add: training error, test error, reconstruction (naive and via pre-image)
def eig(
    fitted_estimator: RealLinalgDecomposition, 
    cross_covariance: Float[Array, "d d"],
    rank: Optional[int] = None,
    ) -> LinalgDecomposition:
    """Eigenvalue decomposition of the fitted Koopman estimator.

    Args:
        fitted_estimator (RealLinalgDecomposition): NamedTuple with the fitted estimator.
        cross_covariance (Float[Array, &quot;d d&quot;]): Cross Covariance matrix.
        rank (Optional[int], optional): Rank of the estimator. Defaults to None meaning full rank. When rank is None, reduced_rank_regression and tikhonov_regression are equivalent and correspond to the EDMD estimator.

    Returns:
        LinalgDecomposition: Eigenvalue decomposition of the Koopman operator.
    """    
    if rank is not None:
        _, idxs = jax.lax.top_k(fitted_estimator.values, rank)
        U = (fitted_estimator.vectors)[:, idxs]
    else:
        U = fitted_estimator.vectors
    
    CPU_DEVICE = jax.devices('cpu')[0]
    #Moving arrays on the CPU as the nonsymmetric eig is not implemented on GPU
    U = jax.device_put(U, CPU_DEVICE)
    _xcov = jax.device_put(cross_covariance, CPU_DEVICE)
    #U@(U.T)@Tw = v w -> (U.T)@T@Uq = vq and w = Uq 
    values, Q = jnp.linalg.eig((U.T)@(_xcov@U))
    return LinalgDecomposition(values, U@Q)

@functools.partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
def sq_error_batch(
    input_features: Float[Array, "d"],
    output_features: Float[Array, "d"],
    estimator: Complex[Array, "d d"]
    ) -> float:
    err = output_features - jnp.dot(estimator.T, input_features)
    return jax.lax.square(jnp.linalg.norm(err))

def sq_error(
    input_data: Float[Array, "n d"],
    output_data: Float[Array, "n d"],
    estimator: Complex[Array, "d d"]
    ) -> float:
    """Mean squared error of the estimator. 

    Args:
        input_data (Float[Array, &quot;n d&quot;]): Initial data
        output_data (Float[Array, &quot;n d&quot;]): Evolved data
        estimator (Complex[Array, &quot;d d&quot;]): Estimator

    Returns:
        float: Mean squared error.
    """    
    return jnp.mean(sq_error_batch(input_data, output_data, estimator))

def naive_predict(
    featurized_x: Float[Array, "d"],
    fitted_estimator: RealLinalgDecomposition, 
    input_data: Float[Array, "n d"],
    output_raw_data: Float[Array, "n dim"],
    rank: Optional[int] = None,
    ) -> Float[Array, "dim"]:

    if rank is not None:
        _, idxs = jax.lax.top_k(fitted_estimator.values, rank)
        U = (fitted_estimator.vectors)[:, idxs]
    else:
        U = fitted_estimator.vectors
    x = einsum(
        featurized_x,
        U,
        U.T,
        input_data,
        output_raw_data,
        "d_in, d_in r, r d_out, n d_out, n dim -> dim"
    )
    num_data = float(input_data.shape[0])
    return jax.lax.reciprocal(num_data)*x