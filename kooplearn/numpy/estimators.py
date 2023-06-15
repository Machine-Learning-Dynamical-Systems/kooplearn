# TO BE MOVED, contrains numpy implementation of the koopman package

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

import numpy as np

from scipy.linalg import eig

# Net goes from X array to X' array

def get_cov(X,Y):
    C = np.cov(X,Y)
    d = X.shape[0]
    # C_X, then X_Y, then C_XY
    return C[:d, :d], C[d:, d:], C[:d, d:]

class ReducedRankRegressor_v2(BaseEstimator, RegressorMixin):

    def set_feature_map(self, phi):
        self.phi = phi

    def fit(self, X, Y, invert_thresh=1e-15):
        """
        fits the Koopman operator given the in memory feature map and datapoints (X_i, Y_i). Results of the fit are stored but not returned.
        Args:
            X (float numpy array):
            Y (float numpy array):
            invert_tresh (float): during matrix inversion, the eigenvalues below this threshold will be treated as zero.
        """
        self.X_ = X
        self.Y_ = Y

        # pass X and Y through feature map, then compute covariance
        C_X, C_Y, C_XY = get_cov(self.phi(X),self.phi(Y))
        self.C_X_ = C_X
        self.C_Y_ = C_Y
        self.C_XY_ = C_XY

        # Computing the koopman operator by inverting C_X
        eigvalX, eigvecX = np.linalg.eig(self.C_X_)
        good_values = eigvalX > invert_thresh
        inv_eigvalX = np.linalg.pinv(np.diag(eig*good_values))

        self.inv_CX_ = eigvecX @ inv_eigvalX @ eigvecX.T

        self.G = self.inv_CX_ @ self.C_XY_

        self.vh_, self.U_, self.V_ = eig(self.G, left=True, right=True)

    def forecast(self, X0, t=1, f = lambda x : x):
        """ 
        Forecast an observable using the estimated Koopman operator.

        Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.
        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1.
            observable (lambda function, optional): Observable to forecast. Defaults to the identity map, corresponding to forecasting the state itself.
            which (None or array of integers, optional): If None, compute the forecast with all the modes of the observable. If which is an array of integers, the forecast is computed using only the modes corresponding to the indexes provided. The modes are arranged in decreasing order with respect to the eigenvalues. For example, if which = [0,2] only the first and third leading modes are used to forecast.  Defaults to None.
        Returns:
            ndarray: Array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """
        check_is_fitted(self, ['vh_', 'U_', 'V_'])
        vh_tm1 = np.diag(np.power(self.vh, t-1))
        f_y = f(self.Y_)
        right = 1/X0.shape * self.phi(X0) @ f_y
        forecasted = self.U_ @ vh_tm1 @ self.V_ @ self.inv_CX_ @ right
        return np.real(forecasted)
    
    def predict(self, X0, f = lambda x : x):
        return self.forecast(X0, t=1, f=f)