"""Nystroem-accelerated Kernel model for Koopman/Transfer operator learning."""

# Authors: The kooplearn developers
# SPDX-License-Identifier: MIT

import logging
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kooplearn.kernel import _regressors
from kooplearn.structs import DynamicalModes

logger = logging.getLogger("kooplearn")


class NystroemKernelRidge(BaseEstimator):
    r"""Nystroem-accelerated Kernel model minimizing the :math:`L^{2}` loss.
    Implements a model approximating the Koopman (deterministic systems) or
    Transfer (stochastic systems) operator by lifting the state with a
    *infinite-dimensional nonlinear* feature map associated to a kernel
    :math:`k` and then minimizing the :math:`L^{2}` loss in the embedded space
    as described in :cite:t:`Meanti2023`.

    .. tip::
        The dynamical modes obtained by calling
        :class:`kooplearn.kernel.NystroemKernelRidge.dynamical_modes` correspond to the *Kernel
        Dynamical Mode Decomposition* by :cite:t:`Williams2015_KDMD`.


    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to retain. If None, all components are used.

    lag_time : int, default=1
        Time delay between the pairs of snapshots `(X_t, X_{t + lag_time})` used to train the estimator.
       
    reduced_rank : bool, default=True
        Whether to use reduced-rank regression introduced in
        :cite:t:`Kostic2022`. If ``False``, initializes the classical
        principal component estimator.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine'} \
            or callable, default='linear'
        Kernel function to use, or a callable that returns a Gram matrix.

    gamma : float or None, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.

    degree : float, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict or None, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    alpha : float, default=0.0
        Tikhonov (ridge) regularization coefficient. ``None`` is equivalent to
        ``tikhonov_reg = 0``, and internally calls specialized stable
        algorithms to deal with this specific case.

    eigen_solver : {'auto', 'dense', 'arpack'}, default='auto'
        Solver used to perform the internal SVD calcuations. If `n_components`
        is much less than the number of training samples, randomized (or
        arpack  to a smaller extent) may be more efficient than the dense
        eigensolver.

        auto :
            the solver is selected by a default policy based on n_samples
            (the number of training samples) and `n_components`:
            if the number of components to extract is less than 10 (strict) and
            the number of samples is more than 200 (strict), the 'arpack'
            method is enabled. Otherwise the exact full eigenvalue
            decomposition is computed and optionally truncated afterwards
            ('dense' method).
        dense :
            run exact full eigenvalue decomposition calling the standard
            LAPACK solver via `scipy.linalg.eigh`, and select the components
            by postprocessing.
        arpack :
            run SVD truncated to n_components calling ARPACK solver using
            `scipy.sparse.linalg.eigsh`. It requires strictly
            0 < n_components < n_samples

    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.

    max_iter : int or None, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    n_centers: int or float, default=0.1
        Number of centers to select. If ``n_centers < 1``, selects
        ``int(n_centers * n_samples)`` centers. If ``n_centers >= 1``,
        selects ``int(n_centers)`` centers. Defaults to ``0.1``.

    random_state : int, RandomState instance or None, default=None
        Used when ``eigen_solver`` == 'arpack'. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy_X : bool, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    nystroem_centers_idxs_ :  ndarray of ints
        Indices of the selected centers.

    X_fit_ : ndarray 
        The data used to fit the model. If `copy_X=False`, then `X_fit_` is
        a reference. This attribute is used for the calls to predict and
        transform. Shape ``(n_samples, n_features)``.

    gamma_ : float
        Kernel coefficient for rbf, poly and sigmoid kernels. When `gamma`
        is explicitly provided, this is just the same as `gamma`. When `gamma`
        is `None`, this is the actual value of kernel coefficient.

    kernel_X_ : ndarray 
        Kernel matrix evaluated at the initial states. Shape ``(n_centers, n_centers)``.

    kernel_Y_ : ndarray
        Kernel matrix evaluated at the evolved states. Shape ``(n_centers, n_centers)``.

    kernel_YX_ : ndarray
        Cross-kernel matrix between evolved and initial states. Shape ``(n_centers, n_centers)``.

    U_ : ndarray
        Projection matrix of shape (n_centers, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)`
        (see :cite:t:`Kostic2022`).

    V_ : ndarray
        Projection matrix of shape (n_centers, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :cite:t:`Kostic2022`).


    Examples
    --------
    .. code-block:: python

        >>> from kooplearn.datasets import make_linear_system
        >>> from kooplearn.kernel import NystroemKernelRidge
        >>> import numpy as np
        >>> 
        >>> # Generate a linear system
        >>> A = np.array([[0.9, 0.1], [-0.1, 0.9]])
        >>> X0 = np.array([1.0, 0.0])
        >>> data = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=42).to_numpy()
        >>> 
        >>> # Fit the model
        >>> model = NystroemKernelRidge(n_components=2, kernel='linear', alpha=1e-3)
        >>> model.fit(data)
        >>> 
        >>> # Predict the future state
        >>> pred = model.predict(data)
        >>> 
        >>> # Get the eigenvalues of the Koopman operator
        >>> eigvals = model.eig()
    """

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "lag_time": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "reduced_rank": ["boolean"],
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "cosine"}),
            callable,
        ],
        "gamma": [
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "alpha": [
            [Interval(Real, 0, None, closed="left")],
            None,
        ],
        "eigen_solver": [StrOptions({"auto", "dense", "arpack"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "n_centers": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "copy_X": ["boolean"],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        n_components=None,
        *,
        lag_time=1,
        reduced_rank=True,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=0.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        n_centers=0.1,
        random_state=None,
        copy_X=True,
        n_jobs=None,
    ):
        self.n_components = n_components
        self.lag_time = lag_time
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
        self.n_centers = n_centers
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X

    def fit(self, X, y=None):
        """Fits the Nystroem Kernel model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training trajectory data, where `n_samples` is the number of
            samples and `n_features` is the number of features.

        y : ndarray of shape (n_samples, n_features_out), default=None
            Optional observable to use for the fit. If `None`, the observable is the state itself.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        kernel_Xnys, kernel_Ynys = self._pre_fit_checks(X)
        if y is not None:
            if y.shape[0] == X.shape[0]:
                self.y_ = y
            else:
                raise ValueError(
                    f"y has {y.shape[0]} samples, but X has {X.shape[0]}. "
                    "Both must have the same number of samples."
                )
        else:
            self.y_ = None

        # Adjust number of components
        if self.n_components is None:
            n_components = self.kernel_X_.shape[0]
        else:
            n_components = min(self.kernel_X_.shape[0], self.n_components)

        # Adjust regularization parameter
        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha

        # Choose eigen solver
        if self.eigen_solver == "auto":
            if self.kernel_X_.shape[0] > 200 and n_components < 10:
                eigen_solver = "arpack"
            else:
                eigen_solver = "dense"
        else:
            eigen_solver = self.eigen_solver

        # Compute regression
        if self.reduced_rank:
            fit_result = _regressors.nystroem_reduced_rank(
                self.kernel_X_,
                self.kernel_Y_,
                kernel_Xnys,
                kernel_Ynys,
                alpha,
                n_components,
                eigen_solver,
            )
        else:
            fit_result = _regressors.nystroem_pcr(
                self.kernel_X_,
                self.kernel_Y_,
                kernel_Xnys,
                kernel_Ynys,
                alpha,
                n_components,
                eigen_solver,
            )

        self._fit_result = fit_result
        self.U_, self.V_, self._spectral_biases = fit_result.values()
        self.rank_ = self.U_.shape[1]

        logger.info(f"Fitted {self.__class__.__name__} model.")
        return self

    def predict(self, X, n_steps=1, observable=False):
        r"""Predicts the state or, if the system is stochastic, its expected
        value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t=n_steps`` instants
        given the initial conditions ``X``. If ``observable`` is ``True``,
        returns the analogue quantity for the observable instead.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Initial states for prediction.

        n_steps : int, default=1
            Number of steps in the future to predict (returns the last one).

        observable : bool, default=False
            If true, predict returns the observable at time :math:`t`.

        Returns
        -------
        ndarray
            The predicted (expected) state or observable at time :math:`t`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=self.copy_X)
        X_fit, _ = self._split_trajectory(self.X_fit_)
        X_fit = X_fit[self.nys_centers_idxs_]
        K_Xin_X = self._get_kernel(X, X_fit)

        if observable:
            if self.y_ is not None:
                observable_fit, _ = self._split_trajectory(self.y_)
                observable_fit = observable_fit[self.nys_centers_idxs_]
            else:
                raise ValueError(
                    "Observable should be passed when calling fit as the y parameter."
                )
        else:
            observable_fit = X_fit
        pred = _regressors.predict(
            n_steps,
            self._fit_result,
            self.kernel_YX_,
            K_Xin_X,
            observable_fit,
        )
        return pred

    def risk(self, X=None):
        """Compute the estimator risk.

        Parameters
        ----------
        X : ndarray or None, default=None
            Trajectory used to evaluate the risk. If None, evaluates on
            training data.

        Returns
        -------
        float
            Risk of the estimator.
        """
        check_is_fitted(self)
        if X is not None:
            X = validate_data(self, X, reset=False, copy=self.copy_X)
            if X.shape[0] < 1 + self.lag_time:
                raise ValueError(
                    f"X has only {X.shape[0]} samples, but at least "
                    f"{1 + self.lag_time} are required."
                )
            X_val, Y_val = self._split_trajectory(X)
            X_train, Y_train = self._split_trajectory(self.X_fit_)
            X_train, Y_train = (
                X_train[self.nys_centers_idxs_],
                Y_train[self.nys_centers_idxs_],
            )
            kernel_Yv = self._get_kernel(Y_val)
            kernel_XXv = self._get_kernel(X_train, X_val)
            kernel_YYv = self._get_kernel(Y_train, Y_val)
        else:
            kernel_Yv = self.kernel_Y_
            kernel_XXv = self.kernel_X_
            kernel_YYv = self.kernel_Y_

        return _regressors.estimator_risk(
            self._fit_result,
            kernel_Yv,
            self.kernel_Y_,
            kernel_XXv,
            kernel_YYv,
        )

    def eig(self, eval_left_on=None, eval_right_on=None):
        """Compute the eigendecomposition of the estimator.

        Returns
        -------
        dict
            Dictionary containing eigenvalues and eigenvectors.
        """
        check_is_fitted(self)
        eig_result = _regressors.eig(self._fit_result, self.kernel_X_, self.kernel_YX_)

        X_fit, Y_fit = self._split_trajectory(self.X_fit_)
        X_fit, Y_fit = X_fit[self.nys_centers_idxs_], Y_fit[self.nys_centers_idxs_]
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return eig_result["values"]
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            kernel_Xin_X_or_Y = self._get_kernel(eval_right_on, X_fit)
            return eig_result["values"], _regressors.evaluate_eigenfunction(
                eig_result, "right", kernel_Xin_X_or_Y
            )
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            kernel_Xin_X_or_Y = self._get_kernel(eval_left_on, Y_fit)
            return eig_result["values"], _regressors.evaluate_eigenfunction(
                eig_result, "left", kernel_Xin_X_or_Y
            )
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            kernel_Xin_X_or_Y_left = self._get_kernel(eval_left_on, Y_fit)
            kernel_Xin_X_or_Y_right = self._get_kernel(eval_right_on, X_fit)
            return (
                eig_result["values"],
                _regressors.evaluate_eigenfunction(
                    eig_result, "left", kernel_Xin_X_or_Y_left
                ),
                _regressors.evaluate_eigenfunction(
                    eig_result, "right", kernel_Xin_X_or_Y_right
                ),
            )

    def dynamical_modes(self, X, observable=False) -> DynamicalModes:
        """
        Computes the mode decomposition of arbitrary observables of the
        Koopman/Transfer operator at the states defined by ``X``. If :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :cite:t:`Kostic2022` for more details.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            States at which to evaluate the modes.

        observable : bool, default=false
            If true, modes of the observable are computed.

        Returns
        -------
        DynamicalModes
            See :class:`kooplearn.structs.DynamicalModes`
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=self.copy_X)
        if X.shape[0] < 1 + self.lag_time:
            raise ValueError(
                f"X has only {X.shape[0]} samples, but at least "
                f"{1 + self.lag_time} are required."
            )

        # Compute eigendecomposition
        eig_result = _regressors.eig(self._fit_result, self.kernel_X_, self.kernel_YX_)
        # Evaluate the right eigenfunctions on the input points X, see also self.eig
        X_fit, Y_fit = self._split_trajectory(self.X_fit_)
        X_fit = X_fit[self.nys_centers_idxs_]
        Y_fit = Y_fit[self.nys_centers_idxs_]
        K_Xin_X = self._get_kernel(X, X_fit)
        right_eigenfunctions = _regressors.evaluate_eigenfunction(
            eig_result, "right", K_Xin_X
        )  # [num_initial_conditions, rank]
        # Project the observable onto the left eigenfunctions
        if observable:
            if self.y_ is not None:
                _, observable_fit = self._split_trajectory(self.y_)
                observable_fit = observable_fit[self.nys_centers_idxs_]
            else:
                raise ValueError(
                    "Observable should be passed when calling fit as the y parameter."
                )
        else:
            observable_fit = Y_fit  # [num_training_points, num_features]
        left_projections = (
            (eig_result["left"].T) @ observable_fit / (Y_fit.shape[0] ** 0.5)
        )

        dmd = DynamicalModes(
            eig_result["values"], right_eigenfunctions, left_projections
        )

        return dmd

    def svals(self):
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(n_components,)`.
        """
        check_is_fitted(self)
        return _regressors.svdvals(self._fit_result, self.kernel_X_, self.kernel_Y_)

    def _get_kernel(self, X, Y=None):
        """Compute the pairwise kernel matrix."""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "gamma": self.gamma_,
                "degree": self.degree,
                "coef0": self.coef0,
            }
        if Y is None:
            Y = X
        return pairwise_kernels(
            X,
            Y,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **params,
        )

    def _init_kernels(self, X, Y):
        """Initialize kernel matrices for training."""
        K_X = self._get_kernel(X)
        K_Y = self._get_kernel(Y)
        K_YX = self._get_kernel(Y, X)
        return K_X, K_Y, K_YX

    def _pre_fit_checks(self, X):
        """Perform pre-fit checks and initialize kernel matrices."""
        X = validate_data(self, X, copy=self.copy_X)
        if X.shape[0] < 1 + self.lag_time:
            raise ValueError(
                f"X has only {X.shape[0]} samples, but at least "
                f"{1 + self.lag_time} are required."
            )
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma
        self.X_fit_ = X
        X_fit, Y_fit = self._split_trajectory(X)

        # Perform random center selection
        self.nys_centers_idxs_ = self._center_selection(X_fit.shape[0] - self.lag_time)
        X_nys = X_fit[self.nys_centers_idxs_]
        Y_nys = Y_fit[self.nys_centers_idxs_]

        self.kernel_X_, self.kernel_Y_, self.kernel_YX_ = self._init_kernels(
            X_nys, Y_nys
        )
        kernel_Xnys, kernel_Ynys = self._init_nys_kernels(X_fit, Y_fit, X_nys, Y_nys)

        return (
            kernel_Xnys,
            kernel_Ynys,
        )  # Don't need to store them; they only serve the purpose of fitting.

    def _init_nys_kernels(
        self, X: np.ndarray, Y: np.ndarray, X_nys: np.ndarray, Y_nys: np.ndarray
    ):
        K_X = self._get_kernel(X, X_nys)
        K_Y = self._get_kernel(Y, Y_nys)
        return K_X, K_Y

    def _center_selection(self, num_pts: int):
        if self.n_centers < 1:
            n_centers = int(np.ceil(self.n_centers * num_pts))
        else:
            n_centers = int(self.n_centers)

        n_centers = min(n_centers, num_pts)

        rng = np.random.default_rng(self.random_state)
        rand_indices = rng.choice(num_pts, n_centers)
        return rand_indices

    def _split_trajectory(self, X):
        """Split a trajectory into context and target pairs."""
        return X[: -self.lag_time], X[self.lag_time :]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True
        tags.non_deterministic = self.eigen_solver == "randomized"
        return tags
