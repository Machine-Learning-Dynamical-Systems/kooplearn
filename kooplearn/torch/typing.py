from typing import NamedTuple
from torch import Tensor
from jaxtyping import Float, Complex


class LinalgDecomposition(NamedTuple):
    values: Complex[Tensor, "values"]
    vectors: Complex[Tensor, "n vectors"]

class RealLinalgDecomposition(NamedTuple):
    values: Float[Tensor, "values"]
    vectors: Float[Tensor, "n vectors"]