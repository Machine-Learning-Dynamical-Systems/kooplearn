from typing import Literal

import numpy as np
import scipy.linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd

from kooplearn._utils import fuzzy_parse_complex, spd_neg_pow, topk
from kooplearn.kernel.linalg import (
    add_diagonal_,
    eigh_rank_reveal,
    stable_topk,
    weighted_norm,
)
from kooplearn.structs import EigResult, FitResult

__all__ = [
    "primal_pcr",
    "primal_rand_pcr",
    "primal_rand_reduced_rank",
    "primal_reduced_rank",
]


def estimator_risk(
    fit_result: FitResult,
    cov_Xv: np.ndarray,
    cov_Yv: np.ndarray,
    cov_XYv: np.ndarray,
    cov_XY: np.ndarray,
) -> float:
    """
    Estimate validation risk.

    Parameters
    ----------
    cov_Xv : ndarray of shape (n_val, n_val)
        Covariance matrix of validation inputs.
    cov_Yv : ndarray of shape (n_val, n_val)
        Covariance matrix of validation outputs.
    cov_XYv : ndarray of shape (n_train, n_val)
        Cross-covariance matrix of validation data.
    cov_XY : ndarray of shape (n_train, n_train)
        Cross-covariance matrix of training data.
    fit_result : FitResult
        Dictionary containing 'U' and 'V' matrices.

    Returns
    -------
    float
        Estimated validation risk.
    """
    U = fit_result["U"]
    r_Y = np.trace(cov_Yv)
    r_XY = -2 * np.trace(np.linalg.multi_dot([cov_XY.T, U, U.T, cov_XYv]))
    r_X = np.trace(np.linalg.multi_dot([cov_XY.T, U, U.T, cov_Xv, U, U.T, cov_XY]))
    return r_Y + r_XY + r_X


def eig(
    fit_result: FitResult,
    C_XY: np.ndarray,
) -> EigResult:
    # Using the trick described in https://arxiv.org/abs/1905.11490
    U = fit_result["U"]
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = scipy.linalg.eig(M, left=True, right=True)
    values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    l_perm = np.argsort(values.conj())
    values = values[r_perm]

    # Normalization in RKHS norm
    rv = U @ rv
    rv = rv[:, r_perm]
    rv = rv / np.linalg.norm(rv, axis=0)
    # Biorthogonalization
    lv = np.linalg.multi_dot([C_XY.T, U, lv])
    lv = lv[:, l_perm]
    l_norm = np.sum(lv * rv, axis=0)
    lv = lv / l_norm

    result: EigResult = {"values": values, "left": lv, "right": rv}
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    phi_Xin: np.ndarray,
):
    lv_or_rv = eig_result[which]
    return phi_Xin @ lv_or_rv


# def estimator_modes(
#         eig_result: EigResult,
#         fit_result: FitResult,
#         phi_X: np.ndarray,  # Feature map evaluated on the training input data
#         phi_Xin: np.ndarray,  # Feature map evaluated on the initial conditions
# ):
#     lv = eig_result["left"]
#     rv = eig_result["right"]
#     U = fit_result["U"]
#     r_dim = phi_X.shape[0] ** -1.0

#     # Initial conditions
#     rv_in = evaluate_eigenfunction(eig_result, "right", phi_Xin).T  # [rank, num_init_conditions]
#     rv_in = (phi_Xin @ rv).T  # [rank, num_init_conditions]
#     # This should be multiplied on the right by the observable evaluated at the output training data
#     lv_obs = np.linalg.multi_dot([r_dim * phi_X, U, lv]).T
#     return (
#         rv_in[:, :, None] * lv_obs[:, None, :],
#     )  # [rank, num_init_conditions, num_training_points]


def estimator_modes(
    eig_result: EigResult,
    fit_result: FitResult,
    phi_X: np.ndarray,  # Feature map evaluated on the training input data
    phi_Xin: np.ndarray,  # Feature map evaluated on the initial conditions
    C_XY,
):
    U = fit_result["U"]
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = scipy.linalg.eig(M, left=True, right=True)
    values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    l_perm = np.argsort(values.conj())
    # values = values[r_perm]

    # Normalization in RKHS norm
    rv = U @ rv
    rv = rv[:, r_perm]
    rv = rv / np.linalg.norm(rv, axis=0)
    # Biorthogonalization
    lv_full = np.linalg.multi_dot([C_XY.T, U, lv])
    lv_full = lv_full[:, l_perm]
    lv = lv[:, l_perm]
    l_norm = np.sum(lv_full * rv, axis=0)
    lv = lv / l_norm
    r_dim = phi_X.shape[0] ** -1.0

    # Initial conditions
    rv_in = (phi_Xin @ rv).T  # [rank, num_init_conditions]
    # This should be multiplied on the right by the observable evaluated at the output training data
    lv_obs = np.linalg.multi_dot([r_dim * phi_X, U, lv]).T
    return rv_in[:, :, None] * lv_obs[:, None, :]
    # [rank, num_init_conditions, num_training_points]


def predict(
    num_steps: int,
    fit_result: FitResult,
    C_XY: np.ndarray,
    phi_Xin: np.ndarray,
    phi_X: np.ndarray,
    obs_train_Y: np.ndarray,
) -> np.ndarray:
    """
    Predicts future observables using the primal form of the fitted operator.

    Parameters
    ----------
    num_steps : int
        Number of forward steps.
    fit_result : dict
        Dictionary containing basis matrix 'U' (eigenfunctions in feature space).
    C_XY : ndarray of shape (N_train, N_train)
        Cross-covariance matrix between input and output feature maps.
    phi_Xin : ndarray of shape (N_eval, d)
        Feature map evaluated on the initial conditions.
    phi_X : ndarray of shape (N_train, d)
        Feature map evaluated on the training input data.
    obs_train_Y : ndarray of shape (N_train, *obs_shape)
        Observables evaluated on the output training data.

    Returns
    -------
    ndarray of shape (N_eval, *obs_shape)
        Predicted observables after `num_steps` steps.
    """
    # G = U U.T C_XY
    # G^n = (U)(U.T C_XY U)^(n-1)(U.T C_XY)
    U = fit_result["U"]
    num_train = phi_X.shape[0]

    # Flatten observables but remember shape
    obs_shape = obs_train_Y.shape[1:]
    obs_flat = obs_train_Y.reshape(num_train, -1)  # (N_train, n_features)

    # Core components
    phi_Xin_dot_U = phi_Xin @ U  # (N_eval, r)
    U_C_XY_U = np.linalg.multi_dot([U.T, C_XY, U])  # (r, r)
    U_phi_X_obs_Y = np.linalg.multi_dot([U.T, phi_X.T, obs_flat]) * (
        num_train**-1
    )  # (r, n_features)

    # Koopman propagation
    M = np.linalg.matrix_power(U_C_XY_U, num_steps - 1)  # (r, r)
    predictions = np.linalg.multi_dot(
        [phi_Xin_dot_U, M, U_phi_X_obs_Y]
    )  # (N_eval, n_features)

    # Restore observable shape
    predictions = predictions.reshape((phi_Xin.shape[0],) + obs_shape)
    return predictions


def svdvals(
    fit_result: FitResult,
    C_XY,
):
    U = fit_result["U"]
    M = np.linalg.multi_dot([U, U.T, C_XY])
    return np.linalg.svd(M, compute_uv=False)


def pcr(
    C_X: np.ndarray,  # Input covariance matrix
    tikhonov_reg: float = 0.0,
    rank: int | None = None,
    svd_solver: Literal["arpack", "dense"] = "arpack",
    tol: float = 0,
    max_iter: int | None = None,
) -> FitResult:
    dim = C_X.shape[0]
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    add_diagonal_(C_X, tikhonov_reg)
    if svd_solver == "arpack":
        num_arpack_eigs = min(rank + 5, C_X.shape[0] - 1)
        values, vectors = eigsh(
            C_X, k=num_arpack_eigs, which="LM", maxiter=max_iter, tol=tol
        )
    elif svd_solver == "dense":
        values, vectors = eigh(C_X)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(C_X, -tikhonov_reg)

    # values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    # rsqrt_vals = (np.sqrt(values)) ** -1
    # vectors = vectors[:, stable_values_idxs]
    # vectors = vectors @ np.diag(rsqrt_vals)
    vectors, _, rsqrt_evals = eigh_rank_reveal(values, vectors, rank)
    vectors = vectors @ np.diag(rsqrt_evals)
    result: FitResult = {"U": vectors, "V": vectors, "svals": values}
    return result


def rand_pcr(
    C_X: np.ndarray,  # Input covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    iterated_power: int,  # Number of power iterations
    rng_seed: int | None = None,  # Seed for the random number generator
):
    dim = C_X.shape[0]
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    add_diagonal_(C_X, tikhonov_reg)
    vectors, values, _ = randomized_svd(
        C_X,
        rank,
        n_oversamples=n_oversamples,
        n_iter=iterated_power,
        random_state=rng_seed,
    )
    add_diagonal_(C_X, -tikhonov_reg)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    rsqrt_vals = (np.sqrt(values)) ** -1
    vectors = vectors[:, stable_values_idxs]
    vectors = vectors @ np.diag(rsqrt_vals)
    result: FitResult = {"U": vectors, "V": vectors, "svals": values}
    return result


def _reduced_rank_noreg(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arpack", "dense"] = "arpack",
):
    rsqrt_C_X = spd_neg_pow(C_X, -0.5)
    B = rsqrt_C_X @ C_XY
    _crcov = B @ B.T
    if svd_solver == "arpack":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        num_arpack_eigs = min(rank + 5, C_X.shape[0] - 1)
        values, vectors = eigsh(_crcov, num_arpack_eigs)
    elif svd_solver == "dense":  # 'dense'
        values, vectors = eigh(_crcov)
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    rsqrt_vals = (np.sqrt(values)) ** -1
    vectors = vectors[:, stable_values_idxs]
    vectors = vectors @ np.diag(rsqrt_vals)
    vectors = rsqrt_C_X @ vectors
    result: FitResult = {"U": vectors, "V": vectors, "svals": values}
    return result


def reduced_rank(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0.0
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arpack", "dense"] = "arpack",
):
    if tikhonov_reg == 0.0:
        return _reduced_rank_noreg(C_X, C_XY, rank, svd_solver)
    else:
        add_diagonal_(C_X, tikhonov_reg)
        _crcov = C_XY @ C_XY.T
        if svd_solver == "arpack":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            num_arpack_eigs = min(rank + 5, C_X.shape[0] - 1)
            values, vectors = eigsh(_crcov, num_arpack_eigs, M=C_X)
        elif svd_solver == "dense":  # 'dense'
            values, vectors = eigh(_crcov, C_X)

        values, indices = topk(values, rank)
        vectors = vectors[:, indices]

        _norms = weighted_norm(vectors, C_X)
        add_diagonal_(C_X, -tikhonov_reg)
        vectors = vectors @ np.diag(_norms ** (-1.0))
        result: FitResult = {"U": vectors, "V": vectors, "svals": values}
        return result


def rand_reduced_rank(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    tikhonov_reg: float,
    rank: int,
    n_oversamples: int = 5,
    iterated_power: int = 1,
    rng_seed: int | None = None,
    precomputed_cholesky=None,
):
    rng = np.random.default_rng(rng_seed)
    _crcov = C_XY @ C_XY.T
    rng = np.random.default_rng(rng_seed)
    sketch = rng.standard_normal(size=(C_X.shape[0], rank + n_oversamples))

    add_diagonal_(C_X, tikhonov_reg)
    if precomputed_cholesky is None:
        cholesky_decomposition = scipy.linalg.cho_factor(C_X)
    else:
        cholesky_decomposition = precomputed_cholesky
    add_diagonal_(C_X, -tikhonov_reg)
    for _ in range(iterated_power):
        _tmp_sketch = scipy.linalg.cho_solve(cholesky_decomposition, sketch)
        sketch = _crcov @ _tmp_sketch
        sketch, _ = scipy.linalg.qr(sketch, mode="economic")  # QR re-orthogonalization

    sketch_p = scipy.linalg.cho_solve(cholesky_decomposition, sketch)

    F_0 = sketch_p.T @ sketch
    F_1 = sketch_p.T @ _crcov @ sketch_p

    values, vectors = eigh(F_1, F_0)
    _norms = weighted_norm(vectors, F_0)
    vectors = vectors @ np.diag(_norms ** (-1.0))

    values, indices = topk(values, rank)
    vectors = sketch_p @ vectors[:, indices]
    result: FitResult = {"U": vectors, "V": vectors, "svals": values}
    return result
