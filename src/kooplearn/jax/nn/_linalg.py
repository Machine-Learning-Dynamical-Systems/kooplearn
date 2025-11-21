"""Linear Algebra."""
import jax.numpy as jnp
from jax.typing import ArrayLike


def sqrtmh(A: ArrayLike) -> ArrayLike:
    """Compute the square root of a Symmetric or Hermitian positive definite matrix.

    Args:
        A (ArrayLike): Symmetric or Hermitian positive definite matrix of shape `(N, N)`.

    Returns:
        ArrayLike: The matrix square root of A, of shape `(N, N)`.
    """
    L, Q = jnp.linalg.eigh(A)
    zero = jnp.zeros((), dtype=L.dtype)
    threshold = L.max(-1) * L.shape[-1] * jnp.finfo(L.dtype).eps
    L = jnp.where(L > threshold[..., None], L, zero)  # zero out small components
    return (Q * jnp.sqrt(L)[..., None, :]) @ Q.conj().T
