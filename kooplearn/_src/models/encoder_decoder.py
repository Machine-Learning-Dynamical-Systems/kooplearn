from numpy.typing import ArrayLike
from kooplearn._src.models.abc import TrainableFeatureMap
from kooplearn._src.models.edmd import EDMD


class EncoderModel(EDMD):
    def __init__(self, feature_map: TrainableFeatureMap, rank=5, tikhonov_reg=None, svd_solver='full',
                 iterated_power=1, n_oversamples=5, optimal_sketching=False):
        super().__init__(feature_map, rank, tikhonov_reg, svd_solver, iterated_power, n_oversamples, optimal_sketching)

    def fit(self, X: ArrayLike, Y: ArrayLike, datamodule=None):
        # Fitting the feature map
        if not self.feature_map.is_fitted:
            self.feature_map.initialize()
            self.feature_map.fit(X, Y, datamodule)
        if X is None or Y is None:
            X, Y = self.feature_map.datamodule.train_dataset.get_X_Y_numpy_matrices()
        # Fitting the Koopman operator
        self.pre_fit_checks(X, Y)
        super().fit(X, Y)


# class EncoderDecoderModel(BaseModel):
#     def __init__(self, feature_map: TrainableFeatureMap, decoder, tikhonov_reg=None):
#         self.feature_map = feature_map
#         self.tikhonov_reg = tikhonov_reg
#         self.decoder = decoder
#
#     def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
#         raise NotImplementedError("TODO: Implement")
#
#     def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
#         raise NotImplementedError("TODO: Implement")
#
#     def fit(self, X: ArrayLike, Y: ArrayLike):
#         # Fitting the feature map
#         self.feature_map.fit(X, Y)
#         raise NotImplementedError("TODO: Implement")
