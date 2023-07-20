import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from kooplearn._src.models.abc import BaseModel, TrainableFeatureMap
from kooplearn._src.operator_regression import primal

class EncoderModel(BaseModel):
    def __init__(self, feature_map: TrainableFeatureMap, tikhonov_reg: Optional[float] = None):
        """A generic model for the Koopman Operator based on a trainable encoding of the state and on
        Tikhonov-regularized least squares for the estimation of Koopman in the encoded space

        Args:
            feature_map (TrainableFeatureMap): A trainable feature map. Should accept and return Numpy arrays. I
            tikhonov_reg (float, optional): Tikhonov regularization to apply on the least squares Koopman estimator.
            Defaults to None.
        """
        self.feature_map = feature_map
        self.tikhonov_reg = tikhonov_reg

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        if observables is None:
            _obs = self.Y_fit_
        if callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated "
                "at the Y training points.")

        phi_Xin = self.feature_map(X)
        phi_X = self.feature_map(self.X_fit_)
        return primal.predict(t, self.U_, self.C_XY_, phi_Xin, phi_X, _obs)

    def modes(self, Xin: ArrayLike, observables: Optional[Union[Callable, ArrayLike]] = None):
        if observables is None:
            _obs = self.Y_fit_
        if callable(observables):
            _obs = observables(self.Y_fit_)
        elif isinstance(observables, np.ndarray):
            _obs = observables
        else:
            raise ValueError(
                "observables must be either None, a callable or a Numpy array of the observable evaluated at the "
                "Y training points.")
        
        check_is_fitted(self, ['U_', 'V_', 'K_X_', 'K_YX_', 'X_fit_', 'Y_fit_'])
        phi_X = self.feature_map(self.X_fit_)
        phi_Xin = self.feature_map(Xin)
        _gamma = primal.estimator_modes(self.U_, self.C_XY_, phi_X, phi_Xin)
        return np.squeeze(np.matmul(_gamma, _obs)) # [rank, num_initial_conditions, num_observables]

    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        check_is_fitted(self, ['U_', 'C_XY_'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = primal.estimator_eig(self.U_, self.C_XY_)
            self._eig_cache = (w, vl, vr)
        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                phi_Xin = self.feature_map(eval_right_on)
                return w, primal.evaluate_eigenfunction(phi_Xin, vr)
        else:
            if eval_right_on is None:
                phi_Xin = self.feature_map(eval_right_on)
                return w, primal.evaluate_eigenfunction(phi_Xin, vl)
            else:
                phi_Xin = self.feature_map(eval_right_on)
                return w, primal.evaluate_eigenfunction(phi_Xin, vl), primal.evaluate_eigenfunction(phi_Xin, vr)

    def svd(self):
        check_is_fitted(self, ['U_', 'C_XY_'])
        return primal.svdvals(self.U_, self.C_XY_)

    def _init_covs(self, X: ArrayLike, Y: ArrayLike):
        C_X = self.feature_map.cov(X)
        C_Y = self.feature_map.cov(Y)
        C_XY = self.feature_map.cov(X, Y)
        return C_X, C_Y, C_XY

    def pre_fit_checks(self, X: ArrayLike, Y: ArrayLike):
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        C_X, C_Y, C_XY = self._init_covs(X, Y)

        self.C_X_ = C_X
        self.C_Y_ = C_Y
        self.C_XY_ = C_XY

        self.X_fit_ = X
        self.Y_fit_ = Y
        if hasattr(self, '_eig_cache'):
            del self._eig_cache

    def fit(self, X: ArrayLike, Y: ArrayLike):
        # Fitting the feature map
        if not self.feature_map.is_fitted:
            self.feature_map.fit()
        # Fitting the Koopman operator
        self.pre_fit_checks(X, Y)
        vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, svd_solver='full')
        self.U_ = vectors

class EncoderDecoderModel(BaseModel):
    def __init__(self, feature_map: TrainableFeatureMap, decoder, tikhonov_reg=None):
        self.feature_map = feature_map
        self.tikhonov_reg = tikhonov_reg
        self.decoder = decoder

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
        raise NotImplementedError("TODO: Implement")

    def eig(self, eval_left_on: Optional[ArrayLike] = None, eval_right_on: Optional[ArrayLike] = None):
        raise NotImplementedError("TODO: Implement")

    def fit(self, X: ArrayLike, Y: ArrayLike):
        # Fitting the feature map
        self.feature_map.fit(X, Y)
        raise NotImplementedError("TODO: Implement")
