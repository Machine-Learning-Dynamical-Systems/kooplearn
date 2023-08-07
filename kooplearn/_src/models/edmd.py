from __future__ import annotations
import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from kooplearn._src.models.abc import BaseModel, FeatureMap, IdentityFeatureMap
from kooplearn._src.operator_regression import primal


class EDMD(BaseModel):
    def __init__(self, feature_map: FeatureMap = IdentityFeatureMap(), rank=5, tikhonov_reg=None, svd_solver='full',
                 iterated_power=1, n_oversamples=5, optimal_sketching=False, reduced_rank=False, randomized=False):
        super().__init__(rank, tikhonov_reg, svd_solver, iterated_power, n_oversamples, optimal_sketching)
        self.feature_map = feature_map
        self.reduced_rank = reduced_rank
        self.randomized = randomized
        self.C_XY_ = None
        self.C_Y_ = None
        self.C_X_ = None

    def predict(self, X: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None):
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

        phi_Xin = self.feature_map(X)
        phi_X = self.feature_map(self.X_fit_)
        return primal.predict(t, self.U_, self.C_XY_, phi_Xin, phi_X, _obs)

    def modes(self, Xin: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
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
        return np.squeeze(np.matmul(_gamma, _obs))  # [rank, num_initial_conditions, num_observables]

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        check_is_fitted(self, ['U_', 'C_XY_'])
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = primal.estimator_eig(self.U_, self.C_XY_)
            self._eig_cache = (w, vl, vr)
        if eval_left_on is None and eval_right_on is None:
            return w
        elif eval_left_on is None and eval_right_on is not None:
            phi_Xin = self.feature_map(eval_right_on)
            return w, primal.evaluate_eigenfunction(phi_Xin, vr)
        elif eval_left_on is not None and eval_right_on is None:
            phi_Xin = self.feature_map(eval_right_on)  # TODO: check if this is correct, maybe eval_left_on?
            return w, primal.evaluate_eigenfunction(phi_Xin, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            phi_Xin = self.feature_map(eval_right_on)
            return w, primal.evaluate_eigenfunction(phi_Xin, vl), primal.evaluate_eigenfunction(phi_Xin, vr)

    def svd(self):
        check_is_fitted(self, ['U_', 'C_XY_'])
        return primal.svdvals(self.U_, self.C_XY_)

    def _init_covs(self, X: np.ndarray, Y: np.ndarray):
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

    def _verify_adequacy(self, new_obj: EDMD):
        if hasattr(new_obj, 'kernel'):
            return False
        if not hasattr(new_obj, 'feature_map'):
            return False
        super()._verify_adequacy(new_obj)

    def load(self, filename, change_feature_map=True):
        new_obj = super(BaseModel).load(filename)
        self.C_X_ = new_obj.C_X_.copy()
        self.C_Y_ = new_obj.C_Y_.copy()
        self.C_XY_ = new_obj.C_XY_.copy()
        if change_feature_map:
            assert hasattr(new_obj, "feature_map"), "savefile does not contain an edmd model"
            self.feature_map = new_obj.feature_map

    def fit(self, X, Y):
        self.pre_fit_checks(X, Y)
        if self.reduced_rank:
            if self.randomized:
                vectors = primal.fit_rand_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg,
                                                                           self.rank, self.n_oversamples,
                                                                           self.iterated_power)
            else:
                vectors = primal.fit_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank,
                                                                      self.tikhonov_reg, self.svd_solver)
        else:
            if self.randomized:
                vectors = primal.fit_rand_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, self.rank,
                                                   self.n_oversamples,
                                                   self.iterated_power)
            else:
                vectors = primal.fit_tikhonov(self.C_X_, self.tikhonov_reg, self.rank, self.svd_solver)
        self.U_ = vectors


# class EDMDReducedRank(PrimalRegressor):
#     def fit(self, X, Y):
#         self.pre_fit_checks(X, Y)
#         if self.svd_solver == 'randomized':
#             vectors = primal.fit_rand_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg,
#                                                                        self.rank, self.n_oversamples,
#                                                                        self.iterated_power)
#         else:
#             vectors = primal.fit_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg,
#                                                                   self.svd_solver)
#         self.U_ = vectors
#
#
# class EDMD(PrimalRegressor):
#     def fit(self, X, Y):
#         self.pre_fit_checks(X, Y)
#         if self.svd_solver == 'randomized':
#             vectors = primal.fit_rand_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, self.rank, self.n_oversamples,
#                                                self.iterated_power)
#         else:
#             vectors = primal.fit_tikhonov(self.C_X_, self.tikhonov_reg, self.rank, self.svd_solver)  # maybe working?
#         self.U_ = vectors
