# NAME TO BE CHANGED, contains the skeletton for the new package

class FeatureMap():
    def __init__(self):
        self.phi = lambda x: x
    def __call__(self, X):
        return self.phi(X)
    def fit(self, X,Y):
        pass

class Decoder():
    def __call__(self, X):
        return X

class GeneralModel():
    def set_feature_map(self, phi:FeatureMap):
        self.phi_ = phi

    def set_koopman(self, koop):
        self.koop_ = koop

    def set_decoder(self, decoder):
        self.decoder_ = decoder

    def fit_feature_map(self, X,Y):
        self.phi_.fit(X,Y)

    def fit_koop(self,X,Y):
        Xp, Yp = self.phi_(X), self.phi_(Y)
        self.koop_.fit(Xp,Yp)

    def forecast(self, t, X, f=lambda x:x):
        Xp = self.phi_(X)
        Yp = self.koop_.forecast(t, Xp, f=f)
        Y = self.decoder_(Yp)
        return Y