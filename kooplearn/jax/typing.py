from typing import NamedTuple
from jaxtyping import Float, Complex, Array

class LinalgDecomposition(NamedTuple):
    values: Complex[Array, "values"]
    vectors: Complex[Array, "n vectors"]

class RealLinalgDecomposition(NamedTuple):
    values: Float[Array, "values"]
    vectors: Float[Array, "n vectors"]