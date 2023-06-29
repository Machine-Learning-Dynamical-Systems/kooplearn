import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from _algorithms import primal
from BaseKoopmanEstimator import BaseKoopmanEstimator
from kernels import Linear


class DirectRegressor(BaseKoopmanEstimator):
    def __init__(self, kernel=Linear(), rank=5, tikhonov_reg=None, backend='numpy', svd_solver='full', iterated_power=1, n_oversamples=5, optimal_sketching=False):
        """Reduced Rank Regression Estimator for the Koopman Operator
        Args:
            rank (int, optional): Rank of the estimator. Defaults to 5.
            tikhonov_reg (float, optional): Tikhonov regularization parameter. Defaults to None.
            backend (str, optional): 
                If 'numpy' kernel matrices are formed explicitely and stored as numpy arrays. 
                Defaults to 'numpy'.
            svd_solver (str, optional): 
                If 'full', run exact SVD calling LAPACK solver functions. Warning: 'full' is not compatible with the 'keops' backend.
                If 'arnoldi', run SVD truncated to rank calling ARPACK solver functions.
                If 'randomized', run randomized SVD by the method of [add ref.]  
                Defaults to 'full'.
            iterated_power (int, optional): Number of iterations for the power method computed by svd_solver = 'randomized'. Must be of range :math:`[0, \infty)`. Defaults to 2.
            n_oversamples (int, optional): This parameter is only relevant when svd_solver = 'randomized'. It corresponds to the additional number of random vectors to sample the range of X so as to ensure proper conditioning. Defaults to 10.
            optimal_sketching (bool, optional): Sketching strategy for the randomized solver. If true performs optimal sketching (computaitonally more expensive but more accurate). Defaults to False.
        """
        self.rank = rank
        self.tikhonov_reg = tikhonov_reg
        self.backend = backend
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

    def predict(self, X, t=1, observables=None):
        """Predict an observable using the estimated Koopman operator.
        
        This method with t=1., observable = lambda x: x, and which = None is equivalent to self.predict(X).
        Be aware of the unit of measurements: if the datapoints come from a continuous dynamical system disctretized every dt, the variable t in this function corresponds to the time t' = t*dt  of the continuous dynamical system.
        Args:
            X (ndarray): 2D array of shape (n_samples, n_features) containing the initial conditions.
            observables (ndarray, optional): 2D array of observables computed on previously seen data. If None, uses the test Y dataset instead.
            t (scalar or ndarray, optional): Time(s) to forecast. Defaults to 1.
        Returns:
            ndarray: Array of shape (n_t, n_samples, n_obs) containing the forecast of the observable(s) provided as argument. Here n_samples = len(X), n_obs = len(observable(x)), n_t = len(t) (if t is a scalar n_t = 1).
        """ 
        if observables is None:
            return primal.low_rank_predict(t, self.vectors.T, self.C_XY_, X, self.X_fit_, self.Y_fit_)
        return primal.low_rank_predict(t, self.vectors.T, self.C_XY_, X, self.X_fit_, observables)

    def eig(self):
        check_is_fitted(self, ['U_','C_XY_'])
        return primal.low_rank_eig(self.U_, self.C_XY_)

    def apply_eigfun(self, X):
        _,vectors = self.eig()
        return primal.low_rank_eigfun_eval(X, vectors)

    def svd(self):
        check_is_fitted(self, ['U_', 'C_XY_'])
        return primal.svdvals(self.U_, self.C_XY_)

    def _get_cov(self,X,Y):
        # remember X and Y of shape (n_samples, n_features)
        C = self.kernel.cov(X.T,Y.T)
        d = X.shape[1]
        # C_X, then X_Y, then C_XY
        return C[:d, :d], C[d:, d:], C[:d, d:]

class DirectReducedRank(DirectRegressor):

    def fit(self, X, Y):
        self._check_backend_solver_compatibility()
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        self.C_X_, self.C_Y_, self.C_XY_ = self._get_cov(X, Y)

        self.X_fit_ = X
        self.Y_fit_ = Y

        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.n_oversamples, self.iterated_power)
        else:
            vectors = primal.fit_reduced_rank_regression_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.svd_solver)

        self.vectors = vectors

class DirectPrincipalComponent(DirectRegressor):

    def fit(self, X, Y):
        self._check_backend_solver_compatibility()
        X = np.asarray(check_array(X, order='C', dtype=float, copy=True))
        Y = np.asarray(check_array(Y, order='C', dtype=float, copy=True))
        check_X_y(X, Y, multi_output=True)

        self.C_X_, self.C_Y_, self.C_XY_ = self._get_cov(X, Y)

        self.X_fit_ = X
        self.Y_fit_ = Y

        if self.svd_solver == 'randomized':
            vectors = primal.fit_rand_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.n_oversamples, self.iterated_power)
        else:
            vectors = primal.fit_tikhonov(self.C_X_, self.C_XY_, self.rank, self.tikhonov_reg, self.svd_solver)

        self.vectors = vectors