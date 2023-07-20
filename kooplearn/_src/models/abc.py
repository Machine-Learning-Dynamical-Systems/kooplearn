import abc
from typing import Optional, Union, Callable
from numpy.typing import ArrayLike

# Abstract base classes defining the interface to implement when extending kooplearn

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

class FeatureMap(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: ArrayLike) -> ArrayLike:
        pass

    def cov(self, X: ArrayLike, Y: Optional[ArrayLike] = None):
        phi_X = self.__call__(X)
        if Y is None:
            c = phi_X.T @ phi_X
        else:
            phi_Y = self.__call__(Y)
            c = phi_X.T @ phi_Y
        c *= (X.shape[0]) ** (-1)
        return c

class IdentityFeatureMap(FeatureMap):
    def __call__(self, X: ArrayLike):
        return X

class TrainableFeatureMap(FeatureMap):
    # Trainable feature maps should be callable with numpy arrays and return numpy arrays (see FeatureMap abc). Internally thay can do whatever.
    @abc.abstractmethod
    def fit(self, *a, **kw) -> None:
        pass
    
    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        pass