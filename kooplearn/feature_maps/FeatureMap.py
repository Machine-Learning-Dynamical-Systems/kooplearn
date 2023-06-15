class FeatureMap:
    def __init__(self):
        pass

    def __call__(self, X):
        pass

    def fit(self, X,Y):
        pass

    def initialize(self, koopman_estimator, decoder, dataset):
        pass


class IdentityFeatureMap(FeatureMap):
    def __init__(self):
        super().__init__()
        self.phi = lambda x: x

    def __call__(self, X):
        return self.phi(X)
