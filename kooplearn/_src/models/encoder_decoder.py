from numpy.typing import ArrayLike
from kooplearn._src.models.edmd import PrimalRegressor
from kooplearn._src.operator_regression import primal


class EncoderModel(PrimalRegressor):
    def fit(self, X: ArrayLike, Y: ArrayLike):
        # Fitting the feature map
        if not self.feature_map.is_fitted:
            self.feature_map.fit()
        # Fitting the Koopman operator
        self.pre_fit_checks(X, Y)
        vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, svd_solver='full')
        self.U_ = vectors


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
