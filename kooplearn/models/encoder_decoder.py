from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from kooplearn.models.base import BaseModel


class EncoderModel(BaseModel):
    def __init__(self):
        pass

    def fit(self, X: ArrayLike, Y: ArrayLike):
        pass

    def predict(self, X: ArrayLike, t: int = 1):
        pass

    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
        pass

class EncoderDecoderModel(BaseModel):
    def __init__(self):
        pass

    def fit(self, X: ArrayLike, Y: ArrayLike):
        pass

    def predict(self, X: ArrayLike, t: int = 1):
        pass

    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
        pass