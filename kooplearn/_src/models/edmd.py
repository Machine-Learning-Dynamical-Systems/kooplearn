from os import PathLike
import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
import pickle

from kooplearn._src.models.abc import BaseModel, FeatureMap, IdentityFeatureMap
from kooplearn._src.operator_regression import primal

class PrimalRegressor(BaseModel):
    def __init__(self, feature_map: FeatureMap = IdentityFeatureMap(), rank=5, tikhonov_reg=None, svd_solver='full',
                 iterated_power=1, n_oversamples=5, optimal_sketching=False):
        self.feature_map = feature_map
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):
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
    
    def _verify_adequacy(self, new_obj:BaseModel):
        if hasattr(new_obj, 'kernel'): return False
        if not hasattr(new_obj, 'feature_map'): return False
        if self.rank != new_obj.rank: return False
        if self.tikhonov_reg != new_obj.tikhonov_reg: return False
        if self.svd_solver != new_obj.svd_solver: return False
        if self.iterated_power != new_obj.iterated_power: return False
        if self.n_oversamples != new_obj.n_oversamples: return False
        if self.optimal_sketching != new_obj.optimal_sketching: return False
        return True
    
    def load(self, filename, change_feature_map=True):
        try:
            with open(filename, 'rb+') as file:
                new_obj = pickle.load(file)
        except:
            ('Unable to load {}'.format(filename))

        # verifying coherence of model with regard to kernel methods
        assert self._verify_adequacy(new_obj), "Incoherent or different parameters between models"
        check_is_fitted(new_obj, ['U_', 'V_', 'C_X_', 'C_Y_', 'C_XY_', 'X_fit_', 'Y_fit_'])
        self.U_ = new_obj.U_.copy()
        self.V_ = new_obj.V_.copy()
        self.C_X_ = new_obj.C_X_.copy()
        self.C_Y_ = new_obj.C_Y_.copy()
        self.C_XY_ = new_obj.C_XY_.copy()
        self.X_fit_ = new_obj.X_fit_.copy()
        self.Y_fit_ = new_obj.Y_fit_.copy()

        if change_feature_map:
            assert hasattr(new_obj, "feature_map"), "savefile does not contain an edmd model"
            self.feature_map = new_obj.feature_map

class EDMDReducedRank(PrimalRegressor):
    def fit(self, X, Y):
        self.pre_fit_checks(X, Y)
        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg,
                                                                       self.rank, self.n_oversamples,
                                                                       self.iterated_power)
        else:
            vectors = primal.fit_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg,
                                                                  self.svd_solver)
        self.U_ = vectors


class EDMD(PrimalRegressor):
    def fit(self, X, Y):
        self.pre_fit_checks(X, Y)
        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, self.rank, self.n_oversamples,
                                               self.iterated_power)
        else:
            vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, self.tikhonov_reg, self.rank, self.svd_solver) # NOT WORKING
        self.U_ = vectors
