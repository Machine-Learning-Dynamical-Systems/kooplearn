from __future__ import annotations

import logging
from math import floor
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from kooplearn._src.linalg import cov
from kooplearn._src.operator_regression import primal
from kooplearn._src.operator_regression.utils import parse_observables
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import NotFittedError, ShapeError, check_is_fitted
from kooplearn.abc import BaseModel, FeatureMap
from kooplearn.data import TensorContextDataset
from kooplearn.models.feature_maps import IdentityFeatureMap

logger = logging.getLogger("kooplearn")


class Nonlinear(BaseModel):
    """
    Nonlinear model minimizing the :math:`L^{2}` loss.
    Implements a model approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator by lifting the state with a *nonlinear* feature map and then minimizing the :math:`L^{2}` loss in the embedded space as described in :footcite:t:`Kostic2022`.

    .. tip::

        The dynamical modes obtained by calling :class:`kooplearn.models.Nonlinear.modes` correspond to the *Extended Dynamical Mode Decomposition* by :footcite:t:`Williams2015_EDMD`.

    Args:
        feature_map (callable): Dictionary of functions used for the Nonlinear algorithm. Should be a subclass of ``kooplearn.abc.FeatureMap``.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. ``None`` returns the full rank estimator.
        tikhonov_reg (float): Tikhonov (ridge) regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :footcite:t:`Turri2023`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    .. tip::

        A powerful variation proposed by :footcite:t:`Arbabi2017`, known as Hankel-DMD, evaluates the Koopman/Transfer estimators by stacking consecutive snapshots together in a Hankel matrix. When this model is fitted on context windows of length > 2, the lookback window is automatically set to length ``context_len - 1``. Upon fitting, the whole lookback window is passed through the feature map and the results are then flattened and *concatenated* together, realizing an Hankel-EDMD estimator.

    Attributes:
        data_fit : Training data, of type ``kooplearn.data.TensorContextDataset``.
        cov_X : Covariance matrix of the feature map evaluated at the initial states, that is ``self.data_fit.lookback(self.lookback_len)``.
        cov_Y : Covariance matrix of the feature map evaluated at the evolved states, , that is ``self.data_fit.lookback(self.lookback_len, slide_by = 1)``.
        cov_XY : Cross-covariance matrix between initial and evolved states.
        U : Projection matrix of shape (n_out_features, rank). The Koopman/Transfer operator is approximated as :math:`U U^T \mathrm{cov_{XY}}`.
    """

    def __init__(
        self,
        feature_map: FeatureMap = IdentityFeatureMap(),
        reduced_rank: bool = True,
        rank: Optional[int] = None,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        rng_seed: Optional[int] = None,
    ):
        # Check whether the provided feature map is trainable
        if hasattr(feature_map, "fit"):
            if not feature_map.is_fitted:
                raise NotFittedError(
                    """
                    The provided feature map is not fitted. Please call the fit method before initializing the Nonlinear model.
                    """
                )

        # Perform checks on the input arguments:
        if svd_solver not in ["full", "arnoldi", "randomized"]:
            raise ValueError(
                "Invalid svd_solver. Allowed values are 'full', 'arnoldi' and 'randomized'."
            )
        if svd_solver == "randomized" and iterated_power < 0:
            raise ValueError("Invalid iterated_power. Must be non-negative.")
        if svd_solver == "randomized" and n_oversamples < 0:
            raise ValueError("Invalid n_oversamples. Must be non-negative.")
        self.rng_seed = rng_seed
        self._feature_map = feature_map
        self.rank = rank
        if tikhonov_reg is None:
            self.tikhonov_reg = 0.0
        else:
            self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
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

    def feature_map(self, X: ArrayLike):
        X = np.asanyarray(X)
        if X.shape[1] != self.lookback_len:
            raise ShapeError(
                f"The provided data have a context of ({X.shape[1]}), while it should match the lookback length of the model ({self.lookback_len=})"
            )
        _trail_dims = X.shape[2:]
        _n_samples = X.shape[0]
        new_shape = (_n_samples * self.lookback_len, *_trail_dims)
        # Light wrapper around the true feature map to perform reshapings
        X = X.reshape(new_shape)
        feat_X = self._feature_map(X)
        _trail_dims = feat_X.shape[1:]
        feat_X = feat_X.reshape(_n_samples, self.lookback_len, *_trail_dims)
        return feat_X.reshape(_n_samples, -1)

    def fit(self, data: TensorContextDataset) -> Nonlinear:
        """
        Fits the Nonlinear model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model.

        .. attention::

            The attribute :attr:`lookback_len` will be automatically set to ``data.context_length - 1``. The feature map will be evaluated independently for each snapshot in the lookback window, and the results are concatenated together to form a single feature vector. The pseudo-code of this operation is

            .. code-block:: python

                X = data.lookback(data.context_length - 1)
                n_samples = X.shape[0]
                trailing_dims = X.shape[2:]
                #Stack snapshots in the context window to evaluate the feature map
                X = X.reshape((n_samples*self.lookback_len, *trailing_dims))
                feat_X = self.feature_map(X).reshape(n_samples, self.lookback_len, -1)
                #Flatten and concatenate results
                feat_X = feat_X.reshape(n_samples, -1)

        Args:
            data (TensorContextDataset): Dataset of context windows with tensor entries.

        Returns:
            The fitted estimator.
        """
        self._pre_fit_checks(data)
        if self.reduced_rank:
            if self.svd_solver == "randomized":
                vectors = primal.fit_rand_reduced_rank_regression(
                    self.cov_X,
                    self.cov_XY,
                    self.tikhonov_reg,
                    self.rank,
                    self.n_oversamples,
                    self.iterated_power,
                    self.rng_seed,
                )
            else:
                vectors = primal.fit_reduced_rank_regression(
                    self.cov_X,
                    self.cov_XY,
                    self.tikhonov_reg,
                    self.rank,
                    self.svd_solver,
                )
        else:
            if self.svd_solver == "randomized":
                vectors = primal.fit_rand_principal_component_regression(
                    self.cov_X,
                    self.tikhonov_reg,
                    self.rank,
                    self.n_oversamples,
                    self.iterated_power,
                )
            else:
                vectors = primal.fit_principal_component_regression(
                    self.cov_X, self.tikhonov_reg, self.rank, self.svd_solver
                )

        self.U = vectors
        assert self.U.shape[1] <= self.rank
        if self.U.shape[1] < self.rank:
            logger.warning(
                f"Warning: The fitting algorithm discarded {self.rank - self.U.shape[1]} dimensions of the {self.rank} requested out of numerical instabilities.\nThe rank attribute has been updated to {self.U.shape[1]}.\nConsider decreasing the rank parameter."
            )
            self.rank = self.U.shape[1]

        # Final Checks
        check_is_fitted(
            self, ["U", "cov_XY", "cov_X", "cov_Y", "data_fit", "lookback_len"]
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
            X, Y = data.lookback(self.lookback_len), data.lookback(
                self.lookback_len, slide_by=1
            )
            cov_Xv, cov_Yv, cov_XYv = self._init_covs(X, Y)
        else:
            cov_Xv, cov_Yv, cov_XYv = self.cov_X, self.cov_Y, self.cov_XY

        return primal.estimator_risk(cov_Xv, cov_Yv, cov_XYv, self.cov_XY, self.U)

    def predict(
        self,
        data: TensorContextDataset,
        t: int = 1,
        predict_observables: bool = True,
        reencode_every: int = 0,
    ):
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``data.lookback(self.lookback_len)`` being the lookback slice of ``data``.
        If ``data.observables`` is not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``self.data_fit.observables``, if present. Default to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict``will contain the special key ``__state__`` containing the prediction for the state as well.
        """
        check_is_fitted(
            self, ["U", "cov_XY", "cov_X", "cov_Y", "data_fit", "lookback_len"]
        )

        observables = None
        if predict_observables and hasattr(self.data_fit, "observables"):
            observables = self.data_fit.observables

        parsed_obs, expected_shapes, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit
        )

        phi_Xin = self.feature_map(X_inference)
        phi_X = self.feature_map(X_fit)

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
                obs_pred = primal.predict(t, self.U, self.cov_XY, phi_Xin, phi_X, obs)
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

        check_is_fitted(self, ["U", "cov_XY", "lookback_len"])
        if hasattr(self, "_eig_cache"):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = primal.estimator_eig(self.U, self.cov_XY)
            self._eig_cache = (w, vl, vr)

        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            phi_Xin = self.feature_map(eval_right_on.lookback(self.lookback_len))
            return w, primal.evaluate_eigenfunction(phi_Xin, vr)
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            phi_Xin = self.feature_map(eval_left_on.lookback(self.lookback_len))
            return w, primal.evaluate_eigenfunction(phi_Xin, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            phi_Xin_l = self.feature_map(eval_left_on.lookback(self.lookback_len))
            phi_Xin_r = self.feature_map(eval_right_on.lookback(self.lookback_len))

            return (
                w,
                primal.evaluate_eigenfunction(phi_Xin_l, vl),
                primal.evaluate_eigenfunction(phi_Xin_r, vr),
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
            predict_observables (bool): Return the prediction for the observables in ``self.data_fit.observables``, if present. Default to ``True``.
        Returns:
            (modes, eigenvalues): Modes and corresponding eigenvalues of the system at the states defined by ``data``. The result is composed of arrays with shape matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If ``predict_observables = True`` and ``data.observables != None``, the returned ``dict`` will contain the special key ``__state__`` containing the modes for the state as well.
        """
        check_is_fitted(self, ["U", "cov_XY", "data_fit", "lookback_len"])

        observables = None
        if predict_observables and hasattr(self.data_fit, "observables"):
            observables = self.data_fit.observables

        parsed_obs, expected_shapes, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit
        )

        phi_Xin = self.feature_map(X_inference)
        phi_X = self.feature_map(X_fit)
        _gamma, _eigs = primal.estimator_modes(self.U, self.cov_XY, phi_X, phi_Xin)

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

    def svals(self) -> np.ndarray:
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(rank,)`.
        """
        check_is_fitted(self, ["U", "cov_XY"])
        return primal.svdvals(self.U, self.cov_XY)

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
            Nonlinear: The loaded model.
        """
        return pickle_load(cls, filename)

    def _init_covs(
        self, X: ArrayLike, Y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        X = self.feature_map(X)
        Y = self.feature_map(Y)

        cov_X = cov(X)
        cov_Y = cov(Y)
        cov_XY = cov(X, Y)
        return cov_X, cov_Y, cov_XY

    def _pre_fit_checks(self, data: TensorContextDataset) -> None:
        """Performs pre-fit checks on the training data.

        Initialize the covariance matrices and saves the training data.

        Args:
            data (TensorContextDataset): Dataset of context windows with tensor entries.
        """

        # Save the lookback length
        self._lookback_len = data.context_length - 1
        X_fit, Y_fit = data.lookback(self.lookback_len), data.lookback(
            self.lookback_len, slide_by=1
        )
        self.cov_X, self.cov_Y, self.cov_XY = self._init_covs(X_fit, Y_fit)
        self.data_fit = data

        if self.rank is None:
            self.rank = self.cov_X.shape[0]
            logger.info(f"Rank of the estimator set to {self.rank}")

        if hasattr(self, "_eig_cache"):
            del self._eig_cache


class Linear(Nonlinear):
    """
    Linear model minimizing the :math:`L^{2}` loss.
    Implements a model approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator finding the *linear* operator minimizing the :math:`L^{2}` loss as described in :footcite:t:`Kostic2022`.

    .. tip::

        The dynamical modes obtained by calling :class:`kooplearn.models.Linear.modes` correspond to the *Dynamical Mode Decomposition* by :footcite:t:`Schmid2010`.
    """

    def __init__(
        self,
        reduced_rank: bool = True,
        rank: Optional[int] = 5,
        tikhonov_reg: float = 0,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(
            reduced_rank=reduced_rank,
            rank=rank,
            tikhonov_reg=tikhonov_reg,
            svd_solver=svd_solver,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            rng_seed=rng_seed,
        )
