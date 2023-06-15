# main class, encompasses all kooplearn models

from feature_maps.FeatureMap import FeatureMap
from feature_maps.Decoder import Decoder
from estimators import BaseEstimator

from sklearn.utils.validation import check_is_fitted

class GeneralModel():
    def __init__(self):
        # base behaviour uses default objects, which don't alter the input's representation
        self.phi_ = FeatureMap()
        self.decoder_ = Decoder()
        self.is_koop_fitted = False

    def set_feature_map(self, phi:FeatureMap):
        self.phi_ = phi

    def set_koopman(self, koop:BaseEstimator):
        self.koop_ = koop

    def set_decoder(self, decoder:Decoder):
        self.decoder_ = decoder

    def fit_feature_map(self, X,Y):
        self.phi_.fit(X,Y)

    def fit_koop(self,X,Y):
        assert hasattr(self, 'koop'), 'No koopman operator has been found'
        Xp, Yp = self.phi_(X), self.phi_(Y)
        self.koop_.fit(Xp,Yp)
        self.is_koop_fitted = True

    def forecast(self, t, X, f=lambda x:x):
        Xp = self.phi_(X)
        Yp = self.koop_.forecast(t, Xp, f=f)
        Y = self.decoder_(Yp)
        return Y