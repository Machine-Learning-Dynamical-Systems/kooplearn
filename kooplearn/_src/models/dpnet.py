from typing import Optional, Union, Callable
from numpy.typing import ArrayLike
from kooplearn._src.deep_learning.feature_maps.DPNetFeatureMap import DPNetFeatureMap
from kooplearn._src.models import EncoderModel


class DPNet(EncoderModel):
    def __init__(self, feature_map: DPNetFeatureMap, rank=5, tikhonov_reg=None, svd_solver='full',
                 iterated_power=1, n_oversamples=5, optimal_sketching=False):
        super().__init__(feature_map, rank, tikhonov_reg, svd_solver, iterated_power, n_oversamples, optimal_sketching)
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
