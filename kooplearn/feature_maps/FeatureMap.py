class FeatureMap():
    def __init__(self):
        self.phi = lambda x: x
    def __call__(self, X):
        return self.phi(X)
    def fit(self, X,Y):
        pass