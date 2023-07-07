import abc
from typing import Optional, Union, Callable
from numpy.typing import ArrayLike


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: ArrayLike, Y: ArrayLike):
        pass

    @abc.abstractmethod
    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        pass

    @abc.abstractmethod
    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        pass
