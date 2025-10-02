"""Structs used by the `kernel` algorithms."""

from typing import TypedDict

import numpy as np
from numpy import ndarray


class FitResult(TypedDict):
    """Return type for kernel regressors."""

    U: ndarray
    V: ndarray
    svals: ndarray | None


class EigResult(TypedDict):
    """Return type for eigenvalue decompositions of kernel regressors."""

    values: ndarray
    left: ndarray | None
    right: ndarray


class PredictResult(TypedDict):
    """Return type for predictions of kernel regressors."""

    times: ndarray | None
    state: ndarray | None
    observable: ndarray | None