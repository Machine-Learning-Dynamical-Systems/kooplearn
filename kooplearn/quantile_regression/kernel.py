from kooplearn.models.kernel import KernelDMD
from kooplearn._src.operator_regression.utils import contexts_to_markov_train_states
from kooplearn._src.utils import check_is_fitted
from kooplearn.quantile_regression.utils import compute_quantile_robust
import numpy as np

class QuantileDMD(KernelDMD):
    # Add-On to the LowRankRegressor inherited classes

    def quantile_regression(self, X, fun = lambda x : np.mean(x, axis=-1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_YX', 'data_fit'])

        # recovering original training data
        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, lookback_len=self._lookback_len)

        # recovering elements to rebuild the koopman operator
        T = self.data_fit.shape[0]
        k0T = self.kernel(X, X_fit)/T
        U = self.U
        V = self.V
        M = self.kernel_YX / T
        WT = U @ np.linalg.matrix_power(V.T @ M @ U, t-1) @ V.T # koopman operator iterated on indicator function

        # computing and ordering realisations of f on the training set
        _fYfit = np.apply_along_axis(fun, -1, Y_fit[:,-1])
        candidates = np.argsort(_fYfit)

        # estimating the cdf of the function f on X_t
        cdf = np.array([np.sum((k0T @ WT)[:, candidates[:i]], axis=-1) for i in range(T)]).T

        return compute_quantile_robust(_fYfit[candidates], cdf, alpha=alpha, isotonic=isotonic, rescaling=rescaling)
        

    def expected_shortfall(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'V', 'kernel_X', 'kernel_Y', 'kernel_XY', 'data_fit'])
        
        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, lookback_len=self._lookback_len)

        T = X_fit.shape[0]
        k0T = self.kernel(X.reshape(1, -1), X_fit)/T
        U = self.U
        V = self.V
        M = self.K_YX / T
        WT = U @ np.linalg.matrix_power(V.T @ M @ U, t-1) @ V.T

        _fXfit = fun(X_fit)
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
        k0T = self.kernel(X.reshape(1, -1), X_fit)/T
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