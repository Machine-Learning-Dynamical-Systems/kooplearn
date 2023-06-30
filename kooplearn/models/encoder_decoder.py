import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from kooplearn.models.base import BaseModel
from kooplearn._src import primal
from kooplearn._src.encoding_decoding_utils import TrainableFeatureMap, Decoder

class EncoderModel(BaseModel):
    def __init__(self, feature_map: TrainableFeatureMap, tikhonov_reg: Optional[float] = None):
        """A generic model for the Koopman Operator based on a trainable encoding of the state and on Tikhonov-regularized least squares for the estimation of Koopman in the encoded space

        Args:
            feature_map (TrainableFeatureMap): A trainable feature map. Should accept and return Numpy arrays. I
            tikhonov_reg (float, optional): Tikhonov regularization to apply on the least squares Koopman estimator. Defaults to None.
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
            raise ValueError("observables must be either None, a callable or a Numpy array of the observable evaluated at the Y training points.")
        
        phi_testX = self.feature_map(X)
        phi_trainX = self.feature_map(self.X_fit_)
        return primal.low_rank_predict(t, self.U_, self.C_XY_, phi_testX, phi_trainX, _obs)
    
    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):      
        check_is_fitted(self, ['U_','C_XY_'])
        w, vr  = primal.low_rank_eig(self.U_, self.C_XY_)
        if eval_left_on is None:
            if eval_right_on is None:
                return w
            else:
                phi_textX = self.feature_map(eval_right_on)
                return w, primal.low_rank_eigfun_eval(phi_textX, vr)
        else:
            raise NotImplementedError("Left eigenfunctions are not implemented yet.")

    def svd(self):
        check_is_fitted(self, ['U_', 'C_XY_'])
        return primal.svdvals(self.U_, self.C_XY_)
    
    def _init_covs(self, X: ArrayLike, Y: ArrayLike):
        C_X = self.feature_map.cov(X)
        C_Y = self.feature_map.cov(Y)
        C_XY = self.feature_map.cov(X,Y)
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
    
    def fit(self, X: ArrayLike, Y: ArrayLike):
        #Fitting the feature map
        self.feature_map.fit(X, Y)
        #Fitting the Koopman operator
        self.pre_fit_checks(X, Y)
        _rank = self.C_X_.shape[0]
        vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, _rank, self.tikhonov_reg, 'full')
        self.U_ = vectors

class EncoderDecoderModel(BaseModel):
    def __init__(self, feature_map: TrainableFeatureMap, decoder: Decoder, tikhonov_reg=None):
        self.feature_map = feature_map 
        self.tikhonov_reg = tikhonov_reg
        self.decoder = decoder

    def predict(self, X: ArrayLike, t: int = 1, observables: Optional[Union[Callable, ArrayLike]] = None):    
        raise NotImplementedError("TODO: Implement")
        
    
    def eig(self, eval_left_on: Optional[ArrayLike]=None,  eval_right_on: Optional[ArrayLike]=None):
       raise NotImplementedError("TODO: Implement")
    
    def fit(self, X: ArrayLike, Y: ArrayLike):
        #Fitting the feature map
        self.feature_map.fit(X, Y)
        raise NotImplementedError("TODO: Implement")
