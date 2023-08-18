from numpy.typing import ArrayLike
from kooplearn._src.models.abc import TrainableFeatureMap
from kooplearn._src.models.edmd import EDMD


class EncoderModel(EDMD):
    """Encoder Model

    Encoder model is a special case of EDMD where the feature map is trainable.

    Parameters:
        feature_map: Feature map used for the EDMD algorithm.
        reduced_rank: Whether to use a reduced rank estimator.
        randomized: Whether to use a randomized algorithm.
        rank: Rank of the estimator.
        tikhonov_reg: Tikhonov regularization coefficient.
        svd_solver: SVD solver used. Only considered when not using a randomized algorithm (randomized=False)
         Currently supported: 'arnoldi', 'full'.
        iterated_power: Number of power iterations when using a randomized algorithm (randomized=True).
        n_oversamples: Number of oversamples when using a randomized algorithm (randomized=True).

    Attributes:
        X_fit_: X training data of shape (n_samples, n_features) corresponding to the state at time t.
        Y_fit_: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.
        C_X_: Covariance matrix of the feature map evaluated at X_fit_, shape (n_features, n_features).
        C_Y_: Covariance matrix of the feature map evaluated at Y_fit_, shape (n_features, n_features).
        C_XY_: Cross-covariance matrix of the feature map evaluated at X_fit_ and Y_fit_, shape
        (n_features, n_features).
        U_: TODO add description.
    """
    def __init__(self, feature_map: TrainableFeatureMap, rank=5, tikhonov_reg=0, svd_solver='full',
                 iterated_power=1, n_oversamples=5, optimal_sketching=False, reduced_rank=False, randomized=False):
        super().__init__(feature_map, reduced_rank, randomized, rank, tikhonov_reg, svd_solver, iterated_power,
                         n_oversamples)

    def fit(self, X: ArrayLike, Y: ArrayLike, datamodule=None):
        """Fits the EncoderModel model.

        Use either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model. The feature map is fitted if it is not already fitted. Note that
        for feature maps based on neural networks, it is usually required to pass a datamodule.

        Parameters:
            X: X training data of shape (n_samples, n_features) corresponding to the state at time t.
            Y: Y training data of shape (n_samples, n_features) corresponding to the state at time t+1.
        """
        X = self.X_fit_
        Y = self.Y_fit_
        # Fitting the feature map
        if not self.feature_map.is_fitted:
            self.feature_map.initialize()
            self.feature_map.fit(X, Y, datamodule)
        if X is None or Y is None:
            X, Y = self.feature_map.datamodule.train_dataset.get_numpy_matrices()
        # Fitting the Koopman operator
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
