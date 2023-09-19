from kooplearn._src.models.edmd import ExtendedDMD
import numpy as np

class FinEDMD(ExtendedDMD):
    def quantile_regression(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        pass
    def expected_shortfall(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        pass
    def compute_vol(self, X, w, t=1, stable=True):
        pass