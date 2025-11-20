"""Ridge regressors for Koopman/Transfer operator learning."""

# Authors: The kooplearn developers
# SPDX-License-Identifier: MIT

import logging
from numbers import Integral, Real

import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kooplearn._linalg import covariance
from kooplearn._utils import fuzzy_parse_complex
from kooplearn.linear_model import _regressors
from kooplearn.structs import DynamicalModes

logger = logging.getLogger("kooplearn")


class Ridge(BaseEstimator):
    r"""Linear model minimizing the :math:`L^{2}` loss.
    
    Implements a model approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator by lifting the state with a *nonlinear* feature map and then minimizing the :math:`L^{2}` loss in the embedded space as described in :cite:t:`ridge-Kostic2022`.

    .. tip::

        The dynamical modes obtained by calling :class:`kooplearn.linear_model.Ridge.dynamical_modes` correspond to the *Extended Dynamical Mode Decomposition* by :cite:t:`ridge-Williams2015_EDMD`.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to retain. If ``None``, all components are used.

    lag_time : int, default=1
        Time delay between the pairs of snapshots :math:`(X_t, X_{t + \text{lag_time}})` used to train the estimator.

    reduced_rank : bool, default=True
        Whether to use reduced-rank regression introduced in
        :cite:t:`ridge-Kostic2022`. If ``False``, initializes the classical
        principal component estimator.

    alpha : float or None, default=1e-6
        Tikhonov (ridge) regularization coefficient. ``None`` is equivalent to
        ``alpha = 0``, and internally calls specialized stable
        algorithms to deal with this specific case.

    eigen_solver : {'auto', 'dense', 'arpack', 'randomized'}, default='auto'
        Solver used to perform the internal SVD calculations. If ``n_components``
        is much less than the number of training samples, ``randomized`` (or 
        ``arpack`` to a smaller extent) may be more efficient than the ``dense`` solver.

        auto :
            the solver is selected automatically based on the number of samples and components:
            if the number of components to extract is less than 10 (strict) and
            the number of samples is more than 200 (strict), the ``arpack``
            method is enabled. Otherwise the exact full eigenvalue
            decomposition is computed and optionally truncated afterwards
            (``dense`` method).
        dense :
            run exact full eigenvalue decomposition calling the standard
            LAPACK solver via ``scipy.linalg.eigh``, and select the components
            by postprocessing.
        arpack :
            run SVD truncated to ``n_components`` calling ARPACK solver using
            ``scipy.sparse.linalg.eigsh``. It requires strictly
            ``0 < n_components < n_samples``.
        randomized :
            run randomized SVD as described in :cite:t:`ridge-turri2023randomized`.

    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value is chosen automatically by arpack.

    max_iter : int or None, default=None
        Maximum number of iterations for arpack.
        If ``None``, optimal value is chosen automatically by arpack.

    iterated_power : {'auto'} or int, default='auto'
        Number of iterations for the power method computed by
        ``eigen_solver == 'randomized'``. When ``'auto'``, it is set to 7 when
        ``n_components < 0.1 * min(X.shape)``, otherwise it is set to 4.

    n_oversamples : int, default=5
        Number of oversamples when using a randomized algorithm
        (``eigen_solver == 'randomized'``).

    random_state : int, RandomState instance or None, default=None
        Used when ``eigen_solver`` is ``'arpack'`` or ``'randomized'``. Pass an int
        for reproducible results across multiple function calls.

    copy_X : bool, default=True
        If ``True``, input X is copied and stored by the model in the ``X_fit_``
        attribute. If no further changes will be done to X, setting
        ``copy_X=False`` saves memory by storing a reference.

    Attributes
    ----------
    X_fit_ : ndarray of shape (n_samples, n_features)
        The data used to fit the model. If ``copy_X=False``, then ``X_fit_`` is
        a reference to the original data.

    y_fit_ : ndarray of shape (n_samples, ...) or None
        The observable used to fit the model. If no observable is provided during fitting,
        this attribute is ``None``.

    cov_X_ : ndarray of shape (n_features, n_features)
        Covariance matrix evaluated at the initial states, that is :math:`x_t`.

    cov_Y_ : ndarray of shape (n_features, n_features)
        Covariance matrix evaluated at the evolved states, that is :math:`x_{t+1}`.

    cov_XY_ : ndarray of shape (n_features, n_features)
        Cross-covariance matrix between initial and evolved states.

    U_ : ndarray of shape (n_features, n_components)
        Projection matrix. The evolution operator is approximated as
        :math:`U U^T \mathrm{cov_{XY}}`.

    estimator_ : ndarray of shape (n_features, n_features)
        Least-squares estimator :math:`U U^T \mathrm{cov_{XY}}`.

    rank_ : int
        Effective rank of the fitted estimator.

    Examples
    --------
    .. code-block:: python

        >>> from kooplearn.datasets import make_linear_system
        >>> from kooplearn.linear_model import Ridge
        >>> import numpy as np
        >>>
        >>> # Generate a linear system
        >>> A = np.array([[0.9, 0.1], [-0.1, 0.9]])
        >>> X0 = np.array([1.0, 0.0])
        >>> data = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=42).to_numpy()
        >>>
        >>> # Fit the model
        >>> model = Ridge(n_components=2, alpha=1e-3)
        >>> model = model.fit(data)
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
        "alpha": [
            [Interval(Real, 0, None, closed="left")],
            None,
        ],
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
        "random_state": ["random_state"],
        "copy_X": ["boolean"],
    }

    def __init__(
        self,
        n_components=None,
        *,
        lag_time=1,
        reduced_rank=True,
        alpha=1e-6,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        n_oversamples=5,
        random_state=None,
        copy_X=True,
    ):
        self.n_components = n_components
        self.lag_time = lag_time
        self.reduced_rank = reduced_rank
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.copy_X = copy_X

    def fit(self, X, y=None):
        """
        Fit the linear model using the selected algorithm.

        Depending on the model parameters, this method fits the estimator using
        either a randomized or non-randomized algorithm, and either a full-rank
        or reduced-rank regression approach.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training trajectory data.

        y : ndarray of shape (n_samples, n_features_out), default=None
            Optional observable used for training.
            If ``None``, the observable is assumed to be the state itself.

        Returns
        -------
        self : object
            Fitted model instance.
        """
        self._pre_fit_checks(X)
        if y is not None:
            if y.shape[0] == X.shape[0]:
                self.y_fit_ = y
            else:
                raise ValueError(
                    f"y has {y.shape[0]} samples, but X has {X.shape[0]}. "
                    "Both must have the same number of samples."
                )
        else:
            self.y_fit_ = None

        # Adjust number of components
        if self.n_components is None:
            n_components = self.cov_X_.shape[0]
        else:
            n_components = min(self.cov_X_.shape[0], self.n_components)

        # Choose eigen solver
        if self.eigen_solver == "auto":
            if self.cov_X_.shape[0] > 200 and n_components < 10:
                eigen_solver = "arpack"
            else:
                eigen_solver = "dense"
        else:
            eigen_solver = self.eigen_solver

        # Adjust regularization parameter
        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha

        # Set iterated power
        if self.iterated_power == "auto":
            iterated_power = 7 if n_components < 0.1 * min(X.shape) else 4
        else:
            iterated_power = self.iterated_power

        if self.reduced_rank:
            if eigen_solver == "randomized":
                if alpha == 0.0:
                    raise ValueError(
                        "Tikhonov regularization must be specified when "
                        "solver is randomized."
                    )
                fit_result = _regressors.rand_reduced_rank(
                    self.cov_X_,
                    self.cov_XY_,
                    alpha,
                    n_components,
                    self.n_oversamples,
                    iterated_power,
                    self.random_state,
                )
            else:
                fit_result = _regressors.reduced_rank(
                    self.cov_X_,
                    self.cov_XY_,
                    alpha,
                    n_components,
                    eigen_solver,
                    self.tol,
                    self.max_iter,
                    self.random_state,
                )
        else:
            if eigen_solver == "randomized":
                fit_result = _regressors.rand_pcr(
                    self.cov_X_,
                    alpha,
                    n_components,
                    self.n_oversamples,
                    self.iterated_power,
                    self.random_state,
                )
            else:
                fit_result = _regressors.pcr(
                    self.cov_X_,
                    alpha,
                    n_components,
                    eigen_solver,
                    self.tol,
                    self.max_iter,
                    self.random_state,
                )

        self._fit_result = fit_result
        self.U_, _, self._spectral_biases = fit_result.values()
        self.rank_ = self.U_.shape[1]

        assert self.U_.shape[1] <= n_components
        if self.U_.shape[1] < n_components:
            logger.warning(
                f"Warning: The fitting algorithm discarded {n_components - self.U_.shape[1]} dimensions of the {n_components} requested out of numerical instabilities.\nThe rank attribute has been updated to {self.U_.shape[1]}.\nConsider decreasing the rank parameter."
            )
            n_components = self.U_.shape[1]

        self.estimator_ = self.U_ @ self.U_.T @ self.cov_XY_

        logger.info(f"Fitted {self.__class__.__name__} model.")

        return self

    def predict(self, X, n_steps=1, observable=False) -> np.ndarray:
        r"""
        Predict the system state or its expected value after ``n_steps``.

        Computes the predicted state — or, in the case of a stochastic system,
        the expected value :math:`\mathbb{E}[X_t \mid X_0 = X]` — after
        ``t = n_steps`` time steps given the initial conditions ``X``.
        If ``observable=True``, returns the corresponding predicted observable
        instead of the state.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Initial conditions used for prediction.

        n_steps : int, default=1
            Number of future time steps to predict. Only the final predicted state
            (at time ``t = n_steps``) is returned.

        observable : bool, default=False
            If ``True``, returns the predicted observable at time :math:`t`
            instead of the system state.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Predicted (expected) state or observable at time :math:`t = n_steps`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=self.copy_X)
        X_fit, X_lagged_fit = self._split_trajectory(self.X_fit_)

        if observable:
            if self.y_fit_ is not None:
                _, observable_fit = self._split_trajectory(self.y_fit_)
            else:
                raise ValueError(
                    "Observable should be passed when calling fit as the y parameter."
                )
        else:
            observable_fit = X_lagged_fit
        pred = _regressors.predict(
            n_steps,
            self._fit_result,
            self.cov_XY_,
            X,
            X_fit,
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
                    f"X has only {X.shape[0]} samples, but at least"
                    f"{1 + self.lag_time} are required."
                )
            X_val, Y_val = self._split_trajectory(X)
            cov_Xv, cov_Yv, cov_XYv = self._init_covs(X_val, Y_val)
        else:
            cov_Xv, cov_Yv, cov_XYv = self.cov_X_, self.cov_Y_, self.cov_XY_

        return _regressors.estimator_risk(
            self._fit_result, cov_Xv, cov_Yv, cov_XYv, self.cov_XY_
        )

    def eig(self, eval_left_on=None, eval_right_on=None):
        """
        Compute the eigendecomposition of the learned evolution operator.

        This method returns the eigenvalues of the estimated evolution
        operator, and optionally evaluates the corresponding left and/or right
        eigenfunctions on user-provided data.

        Parameters
        ----------
        eval_left_on : ndarray of shape (n_samples, n_features), optional
            Data points on which to evaluate the **left** eigenfunctions.
            If ``None``, left eigenfunctions are not evaluated.

        eval_right_on : ndarray of shape (n_samples, n_features), optional
            Data points on which to evaluate the **right** eigenfunctions.
            If ``None``, right eigenfunctions are not evaluated.

        Returns
        -------
        eigenvalues : ndarray of shape (n_components,)
            Eigenvalues of the estimated operator.

        left_eigenfunctions : ndarray of shape (n_samples, n_components), optional
            Values of the left eigenfunctions evaluated on ``eval_left_on``.
            Returned only if ``eval_left_on`` is provided.

        right_eigenfunctions : ndarray of shape (n_samples, n_components), optional
            Values of the right eigenfunctions evaluated on ``eval_right_on``.
            Returned only if ``eval_right_on`` is provided.
        """
        check_is_fitted(self)
        eig_result = _regressors.eig(self._fit_result, self.cov_XY_)

        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return eig_result["values"]
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            return eig_result["values"], _regressors.evaluate_eigenfunction(
                eig_result, "right", eval_right_on
            )
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            return eig_result["values"], _regressors.evaluate_eigenfunction(
                eig_result, "left", eval_left_on
            )
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            return (
                eig_result["values"],
                _regressors.evaluate_eigenfunction(eig_result, "left", eval_left_on),
                _regressors.evaluate_eigenfunction(eig_result, "right", eval_right_on),
            )

    def dynamical_modes(self, X, observable=False) -> DynamicalModes:
        """
        Compute the mode decomposition of arbitrary observables of the
        evolution operator at the states defined by ``X``. 
        If :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the evolution operator, for any observable
        :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as:
        :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`.
        See :cite:t:`ridge-Kostic2022` for more details.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            States at which to evaluate the modes.

        observable : bool, default=False
            If ``True``, computes the modes of the observable rather than those
            of the system state.

        Returns
        -------
        DynamicalModes
            Object containing the eigenvalues, modes, and their projections.
            See :class:`kooplearn.structs.DynamicalModes` for details.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=self.copy_X)
        if X.shape[0] < 1 + self.lag_time:
            raise ValueError(
                f"X has only {X.shape[0]} samples, but at least "
                f"{1 + self.lag_time} are required."
            )

        X_fit, Y_fit = self._split_trajectory(self.X_fit_)
        # Compute eigendecomposition
        U = self._fit_result["U"]
        M = np.linalg.multi_dot([U.T, self.cov_XY_, U])
        values, lv, rv = scipy.linalg.eig(M, left=True, right=True)
        values = fuzzy_parse_complex(values)
        r_perm = np.argsort(values)
        l_perm = np.argsort(values.conj())
        values = values[r_perm]
        # Normalization in RKHS norm
        rv = (U @ rv)[:, r_perm]
        rv /= np.linalg.norm(rv, axis=0)
        # Biorthogonalization
        lv_full = np.linalg.multi_dot([self.cov_XY_.T, U, lv])
        lv_full = lv_full[:, l_perm]
        lv = lv[:, l_perm]
        # Compute correct orthogonalization for the left projection
        l_norm = np.sum(lv_full * rv, axis=0)
        lv = lv / l_norm
        # Initial conditions
        right_eigenfunctions = (X @ rv) / X_fit.shape[0]  # [num_init_conditions, rank]
        if observable:
            if self.y_fit_ is not None:
                _, observable_fit = self._split_trajectory(self.y_fit_)
            else:
                raise ValueError(
                    "Observable should be passed when calling fit as the y parameter."
                )
        else:
            observable_fit = Y_fit
        left_projections = (
            np.linalg.multi_dot([X_fit / X_fit.shape[0], U, lv]).T @ observable_fit
        )

        dmd = DynamicalModes(values, right_eigenfunctions, left_projections)
        return dmd

    def score(self, X=None, y=None, n_steps=1, observable=False, metric=r2_score, **metric_kws) -> float:
        """
        Score the model predictions for timestep ``n_steps``.

        Computes the ``metric`` (default is ``sklearn.metrics.r2_score``) evaluated between
        the model predictions at timestep ``n_steps`` and the true system state (or observable,
        if ``observable=True``) at the same timestep.

        Parameters:
            X : ndarray of shape (n_samples, n_features) or None, default=None
                Trajectory used to compute the score. If None, evaluates on
                training data.

            y : ndarray of shape (n_samples, n_features) or None, default=None
                Optional observable used to compute the score.

            n_steps : int, default=1
                Number of future time steps on which to compute the score. Only the
                predictions at the final timestep (``t = n_steps``) are compared
                to the true system state (or observable).

            observable : bool, default=False
                If ``True``, returns the predicted observable at time :math:`t`
                instead of the system state.

            metric: callable (default=r2_score)
                The metric function used to score the model predictions.

            metric_kws: dict
                Optional parameters to pass to the metric function.

        Returns:
            score: float
                Metric function value for the model predictions at the next timestep.
        """
        check_is_fitted(self)

        # Case 1: Using training data
        if X is None:
            X_test, Y_test = self._split_trajectory(self.X_fit_)
            if observable:
                if self.y_fit_ is not None:
                    _, target = self._split_trajectory(self.y_fit_)
                else:
                    raise ValueError(
                        "Cannot score on observable: no training observable was provided during fit."
                    )
            else:
                target = Y_test

        # Case 2: Using provided test data
        else:
            X = validate_data(self, X, reset=False, copy=self.copy_X)
            if X.shape[0] < 1 + self.lag_time:
                raise ValueError(
                    f"X has only {X.shape[0]} samples, but at least "
                    f"{1 + self.lag_time} are required."
                )
            X_test, Y_test = self._split_trajectory(X)
            if observable:
                if y is None:
                    raise ValueError(
                        "When observable=True and X is provided, y must contain the corresponding observable values."
                    )
                if y.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"y has {y.shape[0]} samples, but X has {X.shape[0]}. "
                        "Both must have the same number of samples."
                    )
                _, target = self._split_trajectory(y)
            else:
                target = Y_test

        # Make predictions and align timestamps
        pred = self.predict(X_test, n_steps=n_steps, observable=observable)
        if n_steps > 1:
            target = target[n_steps - 1:]
            pred = pred[: -(n_steps - 1)]
        
        return metric(target, pred, **metric_kws)
    
    def _svals(self):
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(n_components,)`.
        """
        check_is_fitted(self)
        return _regressors.svdvals(self._fit_result, self.cov_XY_)

    def _init_covs(self, X, Y):
        """
        Initializes the covariance matrices `cov_X`, `cov_Y`, and `cov_XY`.

        Args:
            stacked (np.ndarray): Training data of shape ``(n_samples, 2,  *features_shape)``. It should be the result of the function :func:`stack_lookback`.

        Returns:
            A tuple containing:
                - ``cov_X`` (np.ndarray): Covariance matrix of the feature map evaluated at X, shape ``(n_features, n_features)``.
                - ``cov_Y`` (np.ndarray): Covariance matrix of the feature map evaluated at Y, shape ``(n_features, n_features)``.
                - ``cov_XY`` (np.ndarray): Cross-covariance matrix of the feature map evaluated at X and Y, shape ``(n_features, n_features)``.
        """
        cov_X = covariance(X)
        cov_Y = covariance(Y)
        cov_XY = covariance(X, Y)
        return cov_X, cov_Y, cov_XY

    def _pre_fit_checks(self, X):
        """Performs pre-fit checks on the training data.

        Initialize the covariance matrices and saves the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training trajectory data, where `n_samples` is the number of
            samples and `n_features` is the number of features.
        """
        X = validate_data(self, X, copy=self.copy_X)
        if X.shape[0] < 1 + self.lag_time:
            raise ValueError(
                f"X has only {X.shape[0]} samples, but at least "
                f"{1 + self.lag_time} are required."
            )
        self.X_fit_ = X
        X_fit, Y_fit = self._split_trajectory(X)
        self.cov_X_, self.cov_Y_, self.cov_XY_ = self._init_covs(X_fit, Y_fit)

    def _split_trajectory(self, X):
        """Split a trajectory into context and target pairs."""
        return X[: -self.lag_time], X[self.lag_time :]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True
        tags.non_deterministic = self.eigen_solver == "randomized"
        return tags
