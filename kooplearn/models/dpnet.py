from typing import Optional, Union, Callable
from numpy.typing import ArrayLike
from kooplearn._src.dpnet_utils.DPNetFeatureMap import DPNetFeatureMap
from kooplearn.models.base import BaseModel
from kooplearn.models.edmd import PrimalRegressor


class DPNet(BaseModel):
    def __init__(self, feature_map: DPNetFeatureMap, koopman_estimator: PrimalRegressor):
        self.feature_map = feature_map
        self.koopman_estimator = koopman_estimator
        self.datamodule = None

    def fit(self, X, Y, datamodule=None):
        if not self.feature_map.is_fitted:
            self.feature_map.initialize()
            self.feature_map.fit(X, Y, datamodule)
        if X is None or Y is None:
            X, Y = self.feature_map.datamodule.train_dataset.get_X_Y_numpy_matrices()
        self.koopman_estimator.feature_map = self.feature_map
        self.koopman_estimator.fit(X, Y)

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        Y = self.koopman_estimator.predict(X, t, observables)
        return Y

    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        return self.koopman_estimator.eig(eval_left_on, eval_right_on)
