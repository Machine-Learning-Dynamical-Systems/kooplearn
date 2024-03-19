from __future__ import annotations

import logging
from math import floor
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import DotProduct

from kooplearn._src.operator_regression import dual
from kooplearn._src.operator_regression.utils import parse_observables
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import NotFittedError, ShapeError, check_is_fitted
from kooplearn.abc import BaseModel
from kooplearn.data import TensorContextDataset

logger = logging.getLogger("kooplearn")


class Kernel(BaseModel, RegressorMixin):
    """
    Kernel model minimizing the :math:`L^{2}` loss.
    Implements a model approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator by lifting the state with a *infinite-dimensional nonlinear* feature map associated to a kernel :math:`k` and then minimizing the :math:`L^{2}` loss in the embedded space as described in :footcite:t:`Kostic2022`.

    .. tip::

        The dynamical modes obtained by calling :class:`kooplearn.models.Kernel.modes` correspond to the *Kernel Dynamical Mode Decomposition* by :footcite:t:`Williams2015_KDMD`.

    Args:
        kernel (sklearn.gaussian_process.kernels.Kernel): sklearn Kernel object. Defaults to `DotProduct`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. Defaults to 5.
        tikhonov_reg (float): Tikhonov (ridge) regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :footcite:t:`Turri2023`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        optimal_sketching (bool): Sketching strategy for the randomized solver. If `True` performs optimal sketching (computaitonally expensive but more accurate).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    .. tip::

        A powerful DMD variation proposed by :footcite:t:`Arbabi2017`, known as Hankel-DMD, evaluates the Koopman/Transfer estimators by stacking consecutive snapshots together in a Hankel matrix. When this model is fitted context windows of length > 2, the lookback window length is automatically set to ``context_len - 1``. Upon fitting, the whole lookback window is flattened and *concatenated* together before evaluating the kernel function, thus realizing an Hankel-KDMD estimator.

    Attributes:
        data_fit : Training data, of type ``kooplearn.data.TensorContextDataset``.
        kernel_X : Kernel matrix evaluated at the initial states, that is ``self.data_fit[:, :self.lookback_len, ...]`. Shape ``(n_samples, n_samples)``
        kernel_Y : Kernel matrix evaluated at the evolved states, that is ``self.data_fit[:, 1:self.lookback_len + 1, ...]``. Shape ``(n_samples, n_samples)``
        kernel_XY : Cross-kernel matrix between initial and evolved states. Shape ``(n_samples, n_samples)``.
        U : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).
        V : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).

    """

    def __init__(
        self,
        kernel: Kernel = DotProduct(),
        reduced_rank: bool = True,
        rank: int = 5,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "arnoldi",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        optimal_sketching: bool = False,
        rng_seed: Optional[int] = None,
    ):
        # Initial checks
        if svd_solver not in ["full", "arnoldi", "randomized"]:
            raise ValueError(
                "Invalid svd_solver. Allowed values are 'full', 'arnoldi' and 'randomized'."
            )
        if svd_solver == "randomized" and iterated_power < 0:
            raise ValueError("Invalid iterated_power. Must be non-negative.")
        if svd_solver == "randomized" and n_oversamples < 0:
            raise ValueError("Invalid n_oversamples. Must be non-negative.")

        self.rng_seed = rng_seed
        self._kernel = kernel
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer.")

        self.rank = rank
        if tikhonov_reg is None:
            self.tikhonov_reg = 0.0
        else:
            self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

        self.reduced_rank = reduced_rank
        self._is_fitted = False
        self._lookback_len = -1

    @property
    def lookback_len(self) -> int:
        if self._lookback_len < 0:
            raise NotFittedError(
                "The lookback_len attribute is defined upon fitting the model."
            )
        return self._lookback_len

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def kernel(self, X: ArrayLike, Y: Optional[ArrayLike] = None) -> np.ndarray:
        X = np.asanyarray(X)
        X = X.reshape(X.shape[0], -1)
        if Y is not None:
            Y = np.asanyarray(Y)
            Y = Y.reshape(Y.shape[0], -1)
        return self._kernel(X, Y)

    def fit(self, data: TensorContextDataset) -> Kernel:
        """
        Fits the Kernel model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model.


        .. attention::

            The attribute :attr:`lookback_len` will be automatically set to ``data.context_length - 1``. The data will be first flattened along the trailing dimensions before computing the kernel function. The pseudo-code of this operation is

            .. code-block:: python

                X = data.lookback(data.context_length - 1) #Y = data.lookback(data.context_length - 1, slide_by=1)
                #Flatten the trailing dimensions
                X = X.reshape(data.shape[0], -1)
                self.kernel_X = self.kernel(X)

        Args:
             data (TensorContextDataset): Dataset of context windows with tensor entries.

        Returns:
            The fitted estimator.
        """
        self._pre_fit_checks(data)
        if self.reduced_rank:
            if self.svd_solver == "randomized":
                if self.tikhonov_reg == 0.0:
                    raise ValueError(
                        "tikhonov_reg must be specified when solver is randomized."
                    )
                else:
                    U, V, spectral_bias = dual.fit_rand_reduced_rank_regression(
                        self.kernel_X,
                        self.kernel_Y,
                        self.tikhonov_reg,
                        self.rank,
                        self.n_oversamples,
                        self.optimal_sketching,
                        self.iterated_power,
                        rng_seed=self.rng_seed,
                    )
            else:
                U, V, spectral_bias = dual.fit_reduced_rank_regression(
                    self.kernel_X,
                    self.kernel_Y,
                    self.tikhonov_reg,
                    self.rank,
                    self.svd_solver,
                )
        else:
            if self.svd_solver == "randomized":
                U, V, spectral_bias = dual.fit_rand_principal_component_regression(
                    self.kernel_X,
                    self.tikhonov_reg,
                    self.rank,
                    self.n_oversamples,
                    self.iterated_power,
                    rng_seed=self.rng_seed,
                )
            else:
                U, V, spectral_bias = dual.fit_principal_component_regression(
                    self.kernel_X, self.tikhonov_reg, self.rank, self.svd_solver
                )
        self.U = U
        self.V = V
        self.rank = U.shape[1]
        self._spectral_biases = spectral_bias

        # Final Checks
        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "data_fit", "lookback_len"],
        )
        self._is_fitted = True
        logger.info(
            f"Fitted {self.__class__.__name__} model. Lookback length set to {self.lookback_len}"
        )
        return self

    def risk(self, data: Optional[TensorContextDataset] = None) -> float:
        """Risk of the estimator on the validation ``data``.

        Args:
            data (TensorContextDataset or None): Dataset of context windows with tensor entries. If ``None``, evaluates the risk on the training data.

        Returns:
            Risk of the estimator, see Equation 11 of :footcite:p:`Kostic2022` for more details.
        """
        if data is not None:
            if data.context_length != self.data_fit.context_length:
                raise ShapeError(
                    f"The  context length ({data.context_length}) of the validation data does not match the context length of the training data ({self.data_fit.context_length})."
                )
            X_val, Y_val = data.lookback(self.lookback_len), data.lookback(
                self.lookback_len, slide_by=1
            )
            X_train, Y_train = self.data_fit.lookback(
                self.lookback_len
            ), self.data_fit.lookback(self.lookback_len, slide_by=1)

            kernel_Yv = self.kernel(Y_val)
            kernel_XXv = self.kernel(X_train, X_val)
            kernel_YYv = self.kernel(Y_train, Y_val)
        else:
            kernel_Yv = self.kernel_Y
            kernel_XXv = self.kernel_X
            kernel_YYv = self.kernel_Y
        return dual.estimator_risk(
            kernel_Yv, self.kernel_Y, kernel_XXv, kernel_YYv, self.U, self.V
        )

    def predict(
        self,
        data: TensorContextDataset,
        t: int = 1,
        predict_observables: bool = True,
        reencode_every: int = 0,
    ) -> np.ndarray:
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``data.lookback(self.lookback_len)`` being the lookback slice of ``data``.
        If ``data.observables`` is not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``data.observables``, if present. Default to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict``will contain the special key ``__state__`` containing the prediction for the state as well.
        """
        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "data_fit", "lookback_len"],
        )
        observables = None
        if predict_observables and hasattr(data, "observables"):
            observables = data.observables

        parsed_obs, expected_shapes, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit
        )

        K_Xin_X = self.kernel(X_inference, X_fit)

        results = {}
        for obs_name, obs in parsed_obs.items():
            if (reencode_every > 0) and (t > reencode_every):
                if (predict_observables is True) and (observables is not None):
                    raise ValueError(
                        "rencode_every only works when forecasting states, not observables. Consider setting predict_observables to False."
                    )
                else:
                    num_reencodings = floor(t / reencode_every)
                    for k in range(num_reencodings):
                        raise NotImplementedError
            else:
                obs_pred = dual.predict(t, self.U, self.V, self.kernel_YX, K_Xin_X, obs)
                obs_pred = obs_pred.reshape(expected_shapes[obs_name])
                results[obs_name] = obs_pred
        if len(results) == 1:
            return results["__state__"]
        else:
            return results

    def eig(
        self,
        eval_left_on: Optional[TensorContextDataset] = None,
        eval_right_on: Optional[TensorContextDataset] = None,
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (TensorContextDataset or None): Dataset of context windows on which the left eigenfunctions are evaluated.
            eval_right_on (TensorContextDataset or None): Dataset of context windows on which the right eigenfunctions are evaluated.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``  are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``: shape ``(n_samples, rank)``.
        """

        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "lookback_len", "data_fit"],
        )
        if hasattr(self, "_eig_cache"):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
            )
            self._eig_cache = (w, vl, vr)

        X_fit, Y_fit = self.data_fit.lookback(
            self.lookback_len
        ), self.data_fit.lookback(self.lookback_len, slide_by=1)
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            kernel_Xin_X_or_Y = self.kernel(
                eval_right_on.lookback(self.lookback_len), X_fit
            )
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr)
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            kernel_Xin_X_or_Y = self.kernel(
                eval_left_on.lookback(self.lookback_len), Y_fit
            )
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            kernel_Xin_X_or_Y_left = self.kernel(
                eval_left_on.lookback(self.lookback_len), Y_fit
            )
            kernel_Xin_X_or_Y_right = self.kernel(
                eval_right_on.lookback(self.lookback_len), X_fit
            )
            return (
                w,
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_left, vl),
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_right, vr),
            )

    def modes(
        self,
        data: TensorContextDataset,
        predict_observables: bool = True,
    ):
        """
        Computes the mode decomposition of arbitrary observables of the Koopman/Transfer operator at the states defined by ``data``.

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as the initial condition, see the note above.
            predict_observables (bool): Return the prediction for the observables in ``data.observables``, if present. Default to ``True``

        Returns:
            (modes, eigenvalues): Modes and corresponding eigenvalues of the system at the states defined by ``data``. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict`` will contain the special key ``__state__`` containing the modes for the state as well.
        """
        check_is_fitted(
            self, ["U", "V", "kernel_X", "kernel_YX", "lookback_len", "data_fit"]
        )

        observables = None
        if predict_observables and hasattr(data, "observables"):
            observables = data.observables
        parsed_obs, expected_shapes, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit
        )

        if hasattr(self, "_eig_cache"):
            _eigs, lv, rv = self._eig_cache
        else:
            _eigs, lv, rv = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
            )

        K_Xin_X = self.kernel(X_inference, X_fit)
        _gamma = dual.estimator_modes(K_Xin_X, rv, lv)

        results = {}
        for obs_name, obs in parsed_obs.items():
            expected_shape = (self.rank,) + expected_shapes[obs_name]
            res = np.tensordot(_gamma, obs, axes=1).reshape(
                expected_shape
            )  # [rank, num_initial_conditions, ...]
            results[obs_name] = res

        if len(results) == 1:
            return results["__state__"], _eigs
        else:
            return results, _eigs

    def svals(self):
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(rank,)`.
        """
        check_is_fitted(self, ["U", "V", "kernel_X", "kernel_Y"])
        return dual.svdvals(self.U, self.V, self.kernel_X, self.kernel_Y)

    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX = self.kernel(Y, X)
        return K_X, K_Y, K_YX

    def _pre_fit_checks(self, data: TensorContextDataset) -> None:
        """Performs pre-fit checks on the training data.

        Initialize the kernel matrices and saves the training data.

        Args:
            data (TensorContextDataset): Dataset of context windows with tensor entries.
        """
        # Save the lookback length
        self._lookback_len = data.context_length - 1
        X_fit, Y_fit = data.lookback(self.lookback_len), data.lookback(
            self.lookback_len, slide_by=1
        )
        self.kernel_X, self.kernel_Y, self.kernel_YX = self._init_kernels(X_fit, Y_fit)
        self.data_fit = data

        self._check_rank(self.kernel_X.shape[0])

        if hasattr(self, "_eig_cache"):
            del self._eig_cache

    def _check_rank(self, n_samples):
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer.")
        if self.rank > n_samples:
            raise ValueError(
                f"rank must be less than the number of samples ({n_samples})."
            )

    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            Kernel: The loaded model.
        """
        return pickle_load(cls, filename)
