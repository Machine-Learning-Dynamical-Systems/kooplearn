from jaxtyping import Complex, Float, Array

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from kooplearn.jax.typing import RealLinalgDecomposition

@jax.jit
def spd_norm(vec: Complex[Array, "n"], spd_matrix: Float[Array, "n n"]) -> Float:
     _v = jnp.dot(spd_matrix, vec)
     _v_T = jnp.dot(spd_matrix.T, vec)
     return jnp.sqrt(0.5*((jnp.vdot(vec, _v + _v_T)).real))

batch_spd_norm = jax.jit(jax.vmap(spd_norm, in_axes=(1, None), out_axes=0)) # (vecs: Complex[Array, "n r"], spd_matrix: Float[Array, "n n"]) -> Float[Array, "r"]

def generalized_eigh(A: Float[Array, "n n"], B: Float[Array, "n n"]) -> RealLinalgDecomposition:
     #A workaround to solve a real symmetric GEP Av = \lambda Bv problem in JAX. (!! Not numerically efficient)
     Lambda, Q = jnp.linalg.eigh(B)
     rsqrt_Lambda = jnp.diag(jax.lax.rsqrt(Lambda))
     sqrt_B = Q@rsqrt_Lambda
     _A = 0.5*(sqrt_B.T@(A@sqrt_B) + sqrt_B.T@((A.T)@sqrt_B)) #Force Symmetrization
     values, _tmp_vecs = jnp.linalg.eigh(_A) 
     vectors = Q@(rsqrt_Lambda@_tmp_vecs)
     return RealLinalgDecomposition(values, vectors)

def fro_norm(M:Float[Array, 'd d']) -> Float:
     return jnp.linalg.norm(M, ord='fro')

def op_norm(M:Float[Array, 'd d']) -> Float:
     return jnp.linalg.norm(M, ord=2)

def inv_sqrt(M:Float[Array, 'd d'], epsilon:float=1e-6, strategy:str = 'tikhonov') -> Float[Array, 'd d']:
     vals, vecs = jnp.linalg.eigh(M)
     if strategy == 'tikhonov':
          inv_sqrt_vals = jnp.diag((vals + epsilon)**(-0.5))
     elif strategy == 'trunc':     
          inv_sqrt_vals = jnp.diag(jnp.clip(vals, a_min=epsilon)**(-0.5))
     return jnp.linalg.multi_dot([vecs.T, inv_sqrt_vals, vecs]) 