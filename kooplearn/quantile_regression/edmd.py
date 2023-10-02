from kooplearn.models.edmd import ExtendedDMD
from kooplearn.models.deepedmd import DeepEDMD
from kooplearn._src.operator_regression.utils import contexts_to_markov_train_states, contexts_to_markov_predict_states
from kooplearn._src.utils import check_is_fitted
from kooplearn.quantile_regression.utils import compute_quantile_robust
import numpy as np

class FinEDMD(ExtendedDMD):
    def quantile_regression(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'cov_XY', 'cov_X', 'cov_Y', 'data_fit', 'lookback_len'])

        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, lookback_len=self._lookback_len)
        phi_X = self.feature_map(X_fit)

        _fYfit = np.apply_along_axis(fun, -1, Y_fit)
        _fYfit = _fYfit.reshape(_fYfit.shape[0]) # since this is actually a vector
        candidates = np.argsort(_fYfit)

        Xin, _ = contexts_to_markov_predict_states(X, self.lookback_len)
        phi_Xin = self.feature_map(Xin)

        # computing the koopman operator on indicator functions
        num_train = phi_X.shape[0]
        phi_Xin_dot_U = phi_Xin @ self.U
        U_C_XY_U = np.linalg.multi_dot([self.U.T, self.cov_XY, self.U])

        U_dot_phi_X = np.dot(self.U.T, phi_X.T) * (num_train ** -1)

        M = np.linalg.matrix_power(U_C_XY_U, t - 1)
        pred = np.linalg.multi_dot([phi_Xin_dot_U, M, U_dot_phi_X])

        # estimating the cdf of the function f on X_t
        cdf = np.array([np.sum(pred[:, candidates[:i]], axis=-1) for i in range(num_train)]).T
        return compute_quantile_robust(_fYfit[candidates], cdf, alpha=alpha, isotonic=isotonic, rescaling=rescaling)

    def expected_shortfall(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        pass
    def compute_vol(self, X, w, t=1, stable=True):
        return self.predict(X, t, )

class DeepQuantileEDMD(DeepDMD):
    def quantile_regression(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        check_is_fitted(self, ['U', 'cov_XY', 'cov_X', 'cov_Y', 'data_fit', 'lookback_len'])

        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, lookback_len=self._lookback_len)
        phi_X = self.feature_map(X_fit)

        _fYfit = np.apply_along_axis(fun, -1, Y_fit)
        _fYfit = _fYfit.reshape(_fYfit.shape[0]) # since this is actually a vector
        candidates = np.argsort(_fYfit)

        Xin, _ = contexts_to_markov_predict_states(X, self.lookback_len)
        phi_Xin = self.feature_map(Xin)

        # computing the koopman operator on indicator functions
        num_train = phi_X.shape[0]
        phi_Xin_dot_U = phi_Xin @ self.U
        U_C_XY_U = np.linalg.multi_dot([self.U.T, self.cov_XY, self.U])

        U_dot_phi_X = np.dot(self.U.T, phi_X.T) * (num_train ** -1)

        M = np.linalg.matrix_power(U_C_XY_U, t - 1)
        pred = np.linalg.multi_dot([phi_Xin_dot_U, M, U_dot_phi_X])

        # estimating the cdf of the function f on X_t
        cdf = np.array([np.sum(pred[:, candidates[:i]], axis=-1) for i in range(num_train)]).T
        return compute_quantile_robust(_fYfit[candidates], cdf, alpha=alpha, isotonic=isotonic, rescaling=rescaling)

    def expected_shortfall(self, X, fun = lambda x : np.mean(x, axis=1), alpha=0.01, t=1, isotonic=True, rescaling=True):
        pass
    def compute_vol(self, X, w, t=1, stable=True):
        return self.predict(X, t, )