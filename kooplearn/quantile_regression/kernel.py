from kooplearn._src.models.kernel import KernelDMD
from kooplearn._src.utils import check_is_fitted
import numpy as np
from sklearn.isotonic import IsotonicRegression

class FinKernelDMD(KernelDMD):
    # Add-On to the LowRankRegressor inherited classes

    def quantile_regression(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'V', 'K_X', 'K_Y', 'K_YX', 'X_fit', 'Y_fit'])

        T = self.X_fit_.shape[0]
        k0T = self.kernel(X.reshape(1, -1), self.X_fit_)/T
        U = self.U
        V = self.V
        M = self.K_YX / T
        WT = U @ np.linalg.matrix_power(V.T @ M @ U, t-1) @ V.T

        _fXfit = fun(self.X_fit)
        candidates = np.argsort(_fXfit)

        cdf = np.array([np.sum((k0T @ WT)[:, candidates[:i]]) for i in range(T)])
        if isotonic:
            cdf = IsotonicRegression(y_min=0, y_max=cdf.max()).fit_transform(range(T), cdf)
        if rescaling:
            cdf = (cdf - cdf.min())/cdf.max()
        if alpha=='all':
            return _fXfit[candidates], cdf
        for i, level in enumerate(cdf):
            if level >= alpha:
                assert i != 0, 'Not enough data to estimate quantile'
                return _fXfit[candidates[i-1]]
        return _fXfit[candidates[-1]]
        

    def expected_shortfall(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'V', 'K_X', 'K_Y', 'K_YX', 'X_fit', 'Y_fit'])

        T = self.X_fit_.shape[0]
        k0T = self.kernel(X.reshape(1, -1), self.X_fit)/T
        U = self.U
        V = self.V
        M = self.K_YX / T
        WT = U @ np.linalg.matrix_power(V.T @ M @ U, t-1) @ V.T

        _fXfit = fun(self.X_fit)
        candidates = np.argsort(_fXfit)

        cdf = np.array([np.sum((k0T @ WT)[:, candidates[:i]]) for i in range(T)])
        if isotonic:
            cdf = IsotonicRegression(y_min=0, y_max=1).fit_transform(range(T), cdf)
        if rescaling:
            cdf = (cdf - cdf.min())/cdf.max()
        q_index = 0
        for i, level in enumerate(cdf):
            if level >= alpha:
                assert i != 0, 'Not enough data to estimate quantile'
                q_index = i-1
        values_at_risk = _fXfit[candidates[:q_index]]
        return -np.sum(values_at_risk * (k0T @ WT)[:, candidates[:q_index]])/alpha

    def compute_vol(self, X, w, t=1, stable=True):
        """
        X returns at time 0
        w portfolio weights
        """
        T = self.Y_fit_.shape[0]
        k0T = self.kernel(X.reshape(1, -1), self.X_fit)/T
        U = self.U
        V = self.V
        M = self.K_YX / T
        Wt = U @ np.linalg.matrix_power(V.T @ M @ U, t-1) @ V.T

        kwt = k0T @ Wt

        if stable:
            fa = (self.Y_fit @ w - kwt @ self.Y_fit @ w)**2
            return kwt @ fa

        left = kwt*np.eye(T)
        right = kwt.T @ kwt
        return w @ self.Y_fit.T @ (left - right) @ self.Y_fit @ w