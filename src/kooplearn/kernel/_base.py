import logging

from numbers import Integral, Real

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kooplearn.kernel import regressors
from kooplearn.kernel.structs import PredictResult

logger = logging.getLogger("kooplearn")


class Kernel(BaseEstimator):

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "reduced_rank": ["boolean"],
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"}),
            callable,
        ],
        "gamma": [
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "eigen_solver": [StrOptions({"auto", "dense", "arpack", "randomized"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "iterated_power": [
            Interval(Integral, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "n_oversamples": [Interval(Integral, 1, None, closed="left")],
        "optimal_sketching": ["boolean"],
        "random_state": ["random_state"],
        "copy_X": ["boolean"],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        n_components=None,
        *,
        reduced_rank=True,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        n_oversamples=5,
        optimal_sketching=False,
        random_state=None,
        copy_X=True,
        n_jobs=None,
    ):
        self.n_components = n_components
        self.reduced_rank = reduced_rank
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X
        self._is_fitted = False

    def fit(self, X, y=None):
        self._pre_fit_checks(X)

        # adjust n_components according to user inputs
        if self.n_components is None:
            n_components = self.kernel_X_.shape[0]  # use all dimensions
        else:
            n_components = min(self.kernel_X_.shape[0], self.n_components)

        # compute eigenvectors
        if self.eigen_solver == "auto":
            if self.kernel_X_.shape[0] > 200 and n_components < 10:
                eigen_solver = "arpack"
            else:
                eigen_solver = "dense"
        else:
            eigen_solver = self.eigen_solver

        if self.iterated_power == "auto":
            # Checks if the number of iterations is explicitly specified
            # Adjust iterated_power. 7 was found a good compromise for PCA. See #5299
            iterated_power = 7 if n_components < 0.1 * min(X.shape) else 4

        if self.reduced_rank:
            if eigen_solver == "randomized":
                if self.alpha == 0.0:
                    raise ValueError(
                        "tikhonov_reg must be specified when solver is randomized."
                    )
                else:
                    fit_result = regressors.rand_reduced_rank(
                        self.kernel_X_,
                        self.kernel_Y_,
                        self.alpha,
                        n_components,
                        self.n_oversamples,
                        self.optimal_sketching,
                        iterated_power,
                        rng_seed=self.random_state,
                    )
            else:
                fit_result = regressors.reduced_rank(
                    self.kernel_X_,
                    self.kernel_Y_,
                    self.alpha,
                    n_components,
                    eigen_solver,
                )
        else:
            if eigen_solver == "randomized":
                fit_result = regressors.rand_pcr(
                    self.kernel_X_,
                    self.alpha,
                    n_components,
                    self.n_oversamples,
                    iterated_power,
                    rng_seed=self.random_state,
                )
            else:
                fit_result = regressors.pcr(
                    self.kernel_X_,
                    self.alpha, 
                    n_components, 
                    eigen_solver
                )

        # self._fit_result = fit_result
        self.U_, self.V_, self._spectral_biases_ = fit_result.values()
        self.X_fit_ = X
        self._is_fitted = True
        self._predict_result = PredictResult({"times": None, "state": None, "observable": None})

        logger.info(
            f"Fitted {self.__class__.__name__} model."
        )
        return self
    
    def predict(self, X, n_steps=1, observable=None):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=self.copy_X)
        K_Xin_X = self._get_kernel(X, self.X_fit_[:-self.lag_time_])
        # state_pred = regressors.predict(n_steps, self._fit_result, self.kernel_YX_, K_Xin_X, self.X_fit_[:-self.lag_time_])
        state_pred = regressors.predict(n_steps, self.U_, self.V_, self.kernel_YX_, K_Xin_X, self.X_fit_[:-self.lag_time_])

        if observable is not None:
            # obs_pred = regressors.predict(n_steps, self._fit_result, self.kernel_YX_, K_Xin_X, observable[:-self.lag_time_])
            obs_pred = regressors.predict(n_steps, self.U_, self.V_, self.kernel_YX_, K_Xin_X, observable[:-self.lag_time_])
            # self.predict_results_ = PredictResult({"times": n_steps, "state": state_pred, "observable": obs_pred})
            return obs_pred
        else:
            # self.predict_results_ = PredictResult({"times": n_steps, "state": state_pred, "observable": None})
            return state_pred

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _init_kernels(self, X, Y):
        K_X = self._get_kernel(X)
        K_Y = self._get_kernel(Y)
        K_YX = self._get_kernel(Y, X)
        return K_X, K_Y, K_YX

    def _pre_fit_checks(self, X):
        """Performs pre-fit checks on the training data.

        Initialize the kernel matrices and saves the training data.

        Args:
            X 
        """
        X = validate_data(self, X, copy=self.copy_X)
        if X.shape[0] < 2:
            raise ValueError(
                "n_samples=1 is not sufficient to fit the model. "
                "Minimum required samples is 2."
            )
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma
        self.lag_time_ = 1
        X_fit, Y_fit = X[:-self.lag_time_], X[self.lag_time_:]
        self.kernel_X_, self.kernel_Y_, self.kernel_YX_ = self._init_kernels(X_fit, Y_fit)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True
        tags.input_tags.pairwise = self.kernel == "precomputed"
        tags.non_deterministic = self.eigen_solver == "randomized"
        return tags