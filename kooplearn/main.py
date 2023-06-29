# Main abstract class, encompasses all kooplearn models

from feature_maps.FeatureMap import FeatureMap
from feature_maps.Decoder import Decoder
from koopman_estimators.BaseKoopmanEstimator import BaseKoopmanEstimator
from numpy.typing import ArrayLike
from kooplearn.Datasets.TimeseriesDataModule import TimeseriesDataModule


class GeneralModel():
    def __init__(self, feature_map: FeatureMap, decoder: Decoder, koopman_estimator: BaseKoopmanEstimator):
        self.feature_map = feature_map
        self.decoder = decoder
        self.koopman_estimator = koopman_estimator
        self.is_koop_fitted = False

    # Not being used for the moment
    # def set_feature_map(self, phi:FeatureMap):
    #     self.feature_map = phi
    #
    # def set_koopman(self, koop:BaseEstimator):
    #     self.koop_ = koop
    #
    # def set_decoder(self, decoder:Decoder):
    #     self.decoder_ = decoder

    def fit_feature_map(self, x_train: ArrayLike, y_train: ArrayLike, datamodule: TimeseriesDataModule = None):
        self.feature_map.initialize(self.koopman_estimator, self.decoder, datamodule)
        self.feature_map.fit(x_train, y_train)

    def fit_decoder(self, x_train, y_train, datamodule):
        self.decoder.fit(x_train, y_train)

    def fit_koopman_estimator(self, x_train, y_train, datamodule):
        assert hasattr(self, 'koopman_estimator'), 'No koopman operator has been found'
        Xp, Yp = self.feature_map(x_train), self.feature_map(y_train)
        self.koopman_estimator.fit(Xp, Yp)
        self.is_koop_fitted = True

    def fit(self, x_train: ArrayLike, y_train: ArrayLike, datamodule: TimeseriesDataModule = None):
        # Bruno: For the moment it is easier for me to imagine that we can pass a datamodule when working with
        # DNNs, but we can discuss later what kind of data structure we expect as input and output.
        # Anyway, bellow I assure that we can only pass one of the two options and if we pass a datamodule
        # we can also work with x_train and y_train
        if x_train and y_train and datamodule:
            raise ValueError('You cannot pass both a datamodule and x_train, y_train')
        if datamodule:
            datamodule.setup('fit')
            x_train, y_train = datamodule.train_dataset.get_X_Y_numpy_matrices()
        self.fit_feature_map(x_train, y_train, datamodule)
        self.fit_koop(x_train, y_train, datamodule)
        self.fit_decoder(x_train, y_train)

    def forecast(self, t, X, f=lambda x:x):
        Xp = self.feature_map(X)
        Yp = self.koopman_estimator.forecast(t, Xp, f=f)
        Y = self.decoder(Yp)
        return Y
