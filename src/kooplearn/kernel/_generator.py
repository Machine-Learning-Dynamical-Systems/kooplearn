"""Kernel methods for Koopman/Transfer operator learning."""

# Authors: The kooplearn developers
# SPDX-License-Identifier: MIT

import logging
import numpy as np
from numbers import Integral, Real

from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from kooplearn.kernel import _regressors
from kooplearn.structs import DynamicalModes
from kooplearn.kernel import _utils
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
    n_components : int or None, optional
        Number of generator eigenmodes to retain. If ``None``, all components
        are kept.

    reduced_rank : bool, default=True
        If ``True``, use a reduced-rank solver for the Dirichlet regression
        problem. Recommended for large datasets.

    kernel : {'rbf'} or callable, default='rbf'
        Kernel function. Currently only the RBF kernel supports derivative
        computations.

    gamma : float, optional
        RBF kernel scale parameter. If ``None``, defaults to
        ``1 / n_features``.

    degree : float, optional
        Degree for polynomial kernels (not currently supported for generator
        learning).

    coef0 : float, optional
        Offset for polynomial or sigmoid kernels (ignored).

    kernel_params : dict or None, optional
        Additional keyword arguments passed to a user-defined kernel.

    alpha : float or None, default=1e-6
        Tikhonov regularization for the regression problem. If ``None``,
        a specialized unregularized solver is used.

    n_jobs : int, default=1
        Number of parallel workers for kernel computation.

    friction : ndarray of shape (n_features,), optional
        Langevin friction coefficients used in kernel derivative formulas.

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
    This implementation follows:

    *Kostić et al., "Learning the Infinitesimal Generator of Stochastic Diffusion Processes",
    2024.*

    Only RBF kernels currently support derivative-based generator learning.

    Examples
    --------
    >>> model = GeneratorDirichlet(gamma=1.0, friction=np.ones(d))
    >>> model.fit(X)
    >>> eigvals = model.eig()
    >>> f_pred = model.predict(X0, t=1.0, observable=f)
    """

    _parameter_constraints: dict = {
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
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

    }

    def __init__(
        self,
        n_components=None,
        *,
        reduced_rank=True,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1e-6,
        n_jobs=1,
        friction=None,
        shift=1.0,
    ):
        self.n_components = n_components
        self.reduced_rank = reduced_rank
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.n_jobs=n_jobs
        self.friction = friction
        self.shift = shift

    def fit(self, X):
        """
        Fit the Dirichlet-form kernel model to trajectory data.

        This computes:
          - the Gram matrix ``K``,
          - its first and second kernel derivatives,
          - an approximation of the generator via reduced-rank regression,
          - its eigenvalues and eigenfunctions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training states sampled from a diffusion process.

        Returns
        -------
        self : GeneratorDirichlet
            Fitted estimator.
        """
        self._pre_fit_checks(X,self.friction)
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
        if self.reduced_rank:
            self.eigresults = _regressors.reduced_rank_regression_dirichlet(self.kernel_X_,  # noqa: E501
                                                            self.N_,
                                                            self.M_,
                                                            self.shift,
                                                            alpha,
                                                            n_components)
        else:
            raise NotImplementedError

        self.rank_ = self.eigresults["values"].shape[0]

        logger.info(f"Fitted {self.__class__.__name__} model.")
        return self

    def predict(self, X, t, observable,recompute=True) -> ndarray:
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

        modes = self.dynamical_modes(X, np.sqrt(self.shift)*observable, recompute=recompute)
        pred = _regressors.predict_generator(
            t, modes
        )
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
            kernel_Xin_X_or_Y, N_Xin_X_or_Y = self._get_kernel(eval_right_on, self.X_fit_, get_derivatives=True)
            block_matrix = np.block([np.sqrt(self.shift)*kernel_Xin_X_or_Y, N_Xin_X_or_Y])
            return self.eigresults["values"], np.sqrt(2)*_regressors.evaluate_eigenfunction(
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
            kernel_Xin_X_or_Y, N_Xin_X_or_Y = self._get_kernel(eval_right_on, self.X_fit_ , get_derivatives=True)
            block_matrix = np.block([np.sqrt(self.shift)*kernel_Xin_X_or_Y, N_Xin_X_or_Y])
            return (
                self.eigresults["values"],
                _regressors.evaluate_eigenfunction(
                    self.eigresults, "left", kernel_Xin_X_or_Y
                ),
                np.sqrt(2)*_regressors.evaluate_eigenfunction(
                    self.eigresults, "right", block_matrix
                ),
            )

    def dynamical_modes(self, X, observable, recompute=False) -> DynamicalModes:
        """
        Compute the dynamical mode decomposition of an observable.

        For an observable :math:`f`, its expansion in generator modes is:

        .. math::
            f(x) = \sum_{i=1}^r \langle \xi_i, f \rangle \, \psi_i(x),

        where :math:`\xi_i` and :math:`\psi_i` are left and right eigenfunctions.
        Time evolution under the semigroup :math:`e^{t\mathcal{L}}` acts as:

        .. math::
            f_t(x) = \sum_i e^{t \lambda_i} \langle \xi_i, f \rangle \psi_i(x).

        Parameters
        ----------
        X : ndarray
            Points at which the right eigenfunctions will be evaluated.

        observable : ndarray of shape (n_samples, n_dim)
            Observable values at the training points.

        recompute : bool, default=False
            If ``True``, recompute kernel matrices between ``X`` and ``X_fit_``.

        Returns
        -------
        DynamicalModes
            Structured object containing:
            - eigenvalues :math:`e^{\lambda_i}`,
            - mode coefficients,
            - conditioning factors for evolution.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        levecs = self.eigresults["left"]
        npts = levecs.shape[
            0
        ]  # We use the eigenvector to be consistent with the dirichlet estimator that does not have the same shape #obs_train.shape[0]
        if recompute:
            K_Xin_X, N_Xin_X =  self._get_kernel( X, self.X_fit_, get_derivatives=True)
        else:
            K_Xin_X, N_Xin_X = self.kernel_X_, self.N_
        block_matrix = np.block([np.sqrt(self.shift)*K_Xin_X, N_Xin_X])

        conditioning = np.sqrt(2)*_regressors.evaluate_eigenfunction(
            self.eigresults, "right", block_matrix
        )  # [rank, num_initial_conditions]

        modes_ = np.einsum("nr,nd"  + "->rd"  , levecs.conj(), observable ) / np.sqrt(npts)  # [rank, features]
        #modes_ = np.expand_dims(modes_, axis=1)

        result = DynamicalModes(
            np.exp(self.eigresults["values"]), conditioning, modes_
        )
        return result



    def _get_kernel(self, X, Y=None, get_derivatives=False, get_second_derivatives=False):
        """Compute the pairwise kernel matrix."""

        if not(isinstance(self.kernel,str) and self.kernel == "rbf"):
            raise NotImplementedError
        params = {
                "gamma": self.gamma_,
        }
        length_scale = 1/np.sqrt(2 * self.gamma)
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
            N = _utils.return_grad(K_X, X,Y, self.friction, length_scale)
            M = _utils.return_grad2(K_X, X,Y, self.friction, length_scale)
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
            N = _utils.return_grad(K_X, X,Y, self.friction, length_scale)
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
        K_X, M, N = self._get_kernel(X, get_derivatives=True, get_second_derivatives=True)
        return K_X, M, N

    def _pre_fit_checks(self, X, friction):
        """Perform pre-fit checks and initialize kernel matrices."""
        if friction.shape[0] != X.shape[1]:
            raise ValueError
        X = validate_data(self, X)
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma
        self.X_fit_ = X
        self.kernel_X_, self.M_, self.N_ = self._init_kernels(
            self.X_fit_
        )


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True

        return tags
