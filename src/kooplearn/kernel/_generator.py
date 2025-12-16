"""Kernel methods for Koopman/Transfer operator learning."""

# Authors: The kooplearn developers
# SPDX-License-Identifier: MIT

import logging
from numbers import Integral, Real

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kooplearn.kernel import _regressors, _utils
from kooplearn.structs import DynamicalModes

logger = logging.getLogger("kooplearn")


class GeneratorDirichlet(BaseEstimator):
    r"""
    Kernel-based estimator for the infinitesimal generator of diffusion
    processes using the Dirichlet-form method.

    This class approximates the generator :math:`\mathcal{L}` of a stochastic
    differential equation by embedding the data into a reproducing kernel
    Hilbert space (RKHS). The method relies on first and second derivatives
    of the kernel and solves a regularized variational problem equivalent to
    a kernelized Dirichlet-form regression.

    .. math::

        \langle f, -\mathcal{L} g \rangle
        \approx
        f(X)^\top \, (N + M) \, g(X),

    where ``N`` and ``M`` denote kernel first and second derivative blocks.

    Parameters
    ----------
    friction : float or ndarray of shape (n_features,)
        Langevin friction coefficients used in kernel derivative formulas.
    n_components : int or None, optional
        Number of generator eigenmodes to retain. If ``None``, all components
        are kept.

    gamma : float, optional
        RBF kernel scale parameter. If ``None``, defaults to
        ``1 / n_features``.

    alpha : float or None, default=1e-6
        Tikhonov regularization for the regression problem. If ``None``,
        a specialized unregularized solver is used.

    n_jobs : int, default=1
        Number of parallel workers for kernel computation.



    shift : float, default=1.0
        Positive spectral shift applied to improve conditioning of the estimator.

    Attributes
    ----------
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training data used for fitting.

    gamma_ : float
        Effective kernel parameter.

    kernel_X_ : ndarray of shape (n_samples, n_samples)
        Kernel Gram matrix.

    N_ : ndarray
        First kernel derivative block.

    M_ : ndarray
        Second kernel derivative block.

    eigresults : dict
        Result of the regression/eigendecomposition step. Contains entries:

        - ``"values"`` : ndarray of shape (r,)
            Generator eigenvalues.
        - ``"left"`` : ndarray of shape (n_samples, r)
            Left eigenfunctions.
        - ``"right"`` : ndarray
            Right eigenfunctions in RKHS coordinates.

    rank_ : int
        Number of retained eigenmodes.

    Notes
    -----
    This implementation follows :cite:t:`generatordirichlet-kostic2024learning`.

    .. attention::

        Currently, only the RBF kernel is supported for this estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from kooplearn.kernel import GeneratorDirichlet
    >>> # Generate training data from a 2D Ornstein-Uhlenbeck process
    >>> from kooplearn.datasets import make_prinz_potential
    >>> gamma = 1.0
    >>> sigma = 2
    >>> X = make_prinz_potential(X0 = 0, n_steps=int(5e2), gamma=gamma, sigma=sigma)
    >>> model = GeneratorDirichlet(n_components=4, gamma=1.0, friction=np.ones(1,))
    >>> model = model.fit(X)
    >>> eigvals = model.eig()
    >>> f_pred = model.predict(X, t=1.0)
    >>> print(f_pred.shape)
    """

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "kernel": [StrOptions({"rbf"})],
        "gamma": [
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "alpha": [
            [Interval(Real, 0, None, closed="left")],
            None,
        ],
    }

    def __init__(
        self,
        friction,
        n_components=None,
        *,
        gamma=None,
        alpha=1e-6,
        n_jobs=1,
        shift=1.0,
    ):
        self.n_components = n_components
        self.kernel = "rbf"
        self.gamma = gamma
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.friction = friction
        self.shift = shift

    def fit(self, X, y=None):
        """
        Fit the Dirichlet-form kernel model to trajectory data.

        This computes:
          - The Gram matrix ``K``,
          - Its first and second kernel derivatives,
          - An approximation of the generator via reduced-rank regression :cite:t:`generatordirichlet-Kostic2022`,
          - Its eigenvalues and eigenfunctions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training states sampled from a diffusion process.
        y : ndarray of shape (n_samples, n_features_out), default=None
            Optional observable used for training.
            If ``None``, the observable is assumed to be the state itself.

        Returns
        -------
        self : GeneratorDirichlet
            Fitted estimator.
        """
        self._pre_fit_checks(X, self.friction)
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
            n_components = self.kernel_X_.shape[0]
        else:
            n_components = min(self.kernel_X_.shape[0], self.n_components)

        # Choose eigen solver

        # Adjust regularization parameter
        if self.alpha is None:
            alpha = 0.0
        else:
            alpha = self.alpha

        # Compute regression
        self.eigresults = _regressors.reduced_rank_regression_dirichlet(
            self.kernel_X_,
            self.N_,
            self.M_,
            self.shift,
            alpha,
            n_components,
        )

        self.rank_ = self.eigresults["values"].shape[0]

        logger.info(f"Fitted {self.__class__.__name__} model.")
        return self

    def predict(self, X, t, observable=False) -> ndarray:
        r"""
        Predict the expected observable value at time :math:`t`, conditional on
        the initial condition ``X``.

        This computes:

        .. math::
            \mathbb{E}[f(X_t) \mid X_0 = X],

        using the spectral representation of the generator and the Koopman
        semigroup :math:`e^{t \mathcal{L}}`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Evaluation points.

        t : float
            Time horizon for the Koopman propagation.

        observable : bool, default=False
            If ``True``, returns the predicted observable at time :math:`t`
            instead of the system state.


        Returns
        -------
        ndarray of shape (n_samples, n_dim)
            Predicted observable value :math:`\mathbb{E}[f(X_t)]`.
        """
        modes = self.dynamical_modes(X, observable)
        pred = _regressors.predict_generator(t, modes)
        return pred

    def eig(self, eval_left_on=None, eval_right_on=None):
        r"""
        Predict the expected observable value at time :math:`t`, conditional on
        the initial condition ``X``.

        This computes:

        .. math::
            \mathbb{E}[f(X_t) \mid X_0 = X],

        using the spectral representation of the generator and the Koopman
        semigroup :math:`e^{t \mathcal{L}}`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Evaluation points.

        t : float
            Time horizon for the Koopman propagation.

        observable : ndarray of shape (n_samples, n_dim)
            Observable :math:`f(X)` to propagate in time.

        recompute : bool, default=True
            If ``True``, recompute kernel matrices between ``X`` and ``X_fit_``.
            If ``False``, reuse precomputed training kernels.

        Returns
        -------
        ndarray of shape (n_samples, n_dim)
            Predicted observable value :math:`\mathbb{E}[f(X_t)]`.
        """
        check_is_fitted(self)

        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return self.eigresults["values"]
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            eval_right_on = validate_data(self, eval_right_on, reset=False)
            kernel_Xin_X_or_Y, N_Xin_X_or_Y = self._get_kernel(
                eval_right_on, self.X_fit_, get_derivatives=True
            )
            block_matrix = np.block(
                [np.sqrt(self.shift) * kernel_Xin_X_or_Y, N_Xin_X_or_Y]
            )
            return self.eigresults["values"], np.sqrt(
                2
            ) * _regressors.evaluate_eigenfunction(
                self.eigresults, "right", block_matrix
            )
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            eval_left_on = validate_data(self, eval_left_on, reset=False)
            kernel_Xin_X_or_Y = self._get_kernel(eval_left_on, self.X_fit_)
            return self.eigresults["values"], _regressors.evaluate_eigenfunction(
                self.eigresults, "left", kernel_Xin_X_or_Y
            )
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            eval_right_on = validate_data(self, eval_right_on, reset=False)
            eval_left_on = validate_data(self, eval_left_on, reset=False)
            kernel_Xin_X_or_Y, N_Xin_X_or_Y = self._get_kernel(
                eval_right_on, self.X_fit_, get_derivatives=True
            )
            block_matrix = np.block(
                [np.sqrt(self.shift) * kernel_Xin_X_or_Y, N_Xin_X_or_Y]
            )
            return (
                self.eigresults["values"],
                _regressors.evaluate_eigenfunction(
                    self.eigresults, "left", kernel_Xin_X_or_Y
                ),
                np.sqrt(2)
                * _regressors.evaluate_eigenfunction(
                    self.eigresults, "right", block_matrix
                ),
            )

    def dynamical_modes(self, X, observable=False) -> DynamicalModes:
        """
        Compute the dynamical mode decomposition of an observable.

        For an observable :math:`f`, its expansion in generator modes is:

        .. math::
            f(x) = \\sum_{i=1}^r \\langle \\xi_i, f \\rangle \\, \\psi_i(x),

        where :math:`\\xi_i` and :math:`\\psi_i` are left and right eigenfunctions.
        Time evolution under the semigroup :math:`e^{t\\mathcal{L}}` acts as:

        .. math::
            f_t(x) = \\sum_i e^{t \\lambda_i} \\langle \\xi_i, f \\rangle \\psi_i(x).

        Parameters
        ----------
        X : ndarray
            Points at which the right eigenfunctions will be evaluated.

        observable : bool, default=False
            If ``True``, returns the predicted observable at time :math:`t`
            instead of the system state.

        Returns
        -------
        DynamicalModes
            Structured object containing:
            - eigenvalues :math:`e^{\\lambda_i}`,
            - mode coefficients,
            - conditioning factors for evolution.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if observable:
            observable_fit_ = self.y_fit_
        else:
            observable_fit_ = X
        levecs = self.eigresults["left"]
        npts = levecs.shape[
            0
        ]  # We use the eigenvector to be consistent with the dirichlet estimator that does not have the same shape #obs_train.shape[0]

        K_Xin_X, N_Xin_X = self.kernel_X_, self.N_
        block_matrix = np.block([np.sqrt(self.shift) * K_Xin_X, N_Xin_X])

        conditioning = np.sqrt(2) * _regressors.evaluate_eigenfunction(
            self.eigresults, "right", block_matrix
        )  # [rank, num_initial_conditions]

        modes_ = np.einsum(
            "nr,nd" + "->rd", levecs.conj(), np.sqrt(self.shift) * observable_fit_
        ) / np.sqrt(npts)  # [rank, features]
        # modes_ = np.expand_dims(modes_, axis=1)

        result = DynamicalModes(np.exp(self.eigresults["values"]), conditioning, modes_)
        return result

    def _get_kernel(
        self, X, Y=None, get_derivatives=False, get_second_derivatives=False
    ):
        """Compute the pairwise kernel matrix."""

        if not (isinstance(self.kernel, str) and self.kernel == "rbf"):
            raise NotImplementedError
        params = {
            "gamma": self.gamma_,
        }
        length_scale = 1 / np.sqrt(2 * self.gamma_)
        if Y is None:
            Y = X
        if get_derivatives and get_second_derivatives:
            K_X = pairwise_kernels(
                X,
                Y,
                metric=self.kernel,
                filter_params=True,
                n_jobs=self.n_jobs,
                **params,
            )
            N = _utils.return_grad(K_X, X, Y, self.friction, length_scale)
            M = _utils.return_grad2(K_X, X, Y, self.friction, length_scale)
            return K_X, M, N
        elif get_derivatives:
            K_X = pairwise_kernels(
                X,
                Y,
                metric=self.kernel,
                filter_params=True,
                n_jobs=self.n_jobs,
                **params,
            )
            N = _utils.return_grad(K_X, X, Y, self.friction, length_scale)
            return K_X, N
        else:
            return pairwise_kernels(
                X,
                Y,
                metric=self.kernel,
                filter_params=True,
                n_jobs=self.n_jobs,
                **params,
            )

    def _init_kernels(self, X):
        """Initialize kernel matrices for training."""
        K_X, M, N = self._get_kernel(
            X, get_derivatives=True, get_second_derivatives=True
        )
        return K_X, M, N

    def _pre_fit_checks(self, X, friction):
        """Perform pre-fit checks and initialize kernel matrices."""
        if isinstance(friction, float):
            self.friction = np.full(X.shape[1], friction, dtype=float)
        if self.friction.shape[0] != X.shape[1]:
            raise ValueError(
                f"Friction has length {self.friction.shape[0]}, "
                f"but data has {X.shape[1]} features."
            )
        X = validate_data(self, X)
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma
        self.X_fit_ = X
        self.kernel_X_, self.M_, self.N_ = self._init_kernels(self.X_fit_)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True

        return tags
