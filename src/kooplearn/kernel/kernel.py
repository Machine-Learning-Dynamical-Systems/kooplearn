from sklearn.base import BaseEstimator

class Kernel(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method.")