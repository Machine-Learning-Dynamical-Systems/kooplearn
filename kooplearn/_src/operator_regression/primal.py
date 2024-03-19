from typing import Optional

import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd

from kooplearn._src.linalg import eigh_rank_reveal, spd_neg_pow, weighted_norm
from kooplearn._src.utils import fuzzy_parse_complex, topk


def fit_reduced_rank_regression(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0.0
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. Arnoldi is faster for low ranks.
):
    if tikhonov_reg == 0.0:
        return _fit_reduced_rank_regression_noreg(C_X, C_XY, rank, svd_solver)
    else:
        dim = C_X.shape[0]
        reg_input_covariance = C_X + tikhonov_reg * np.identity(dim, dtype=C_X.dtype)
        _crcov = C_XY @ C_XY.T
        if svd_solver == "arnoldi":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            values, vectors = eigsh(_crcov, rank + 3, M=reg_input_covariance)
        else:
            values, vectors = eigh(_crcov, reg_input_covariance)

        top_eigs = topk(values, rank)
        vectors = vectors[:, top_eigs.indices]

        _norms = weighted_norm(vectors, reg_input_covariance)
        vectors = vectors @ np.diag(_norms ** (-1.0))
        return vectors


def _fit_reduced_rank_regression_noreg(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. Arnoldi is faster for low ranks.
):
    rsqrt_C_X = spd_neg_pow(C_X, -0.5)
    B = rsqrt_C_X @ C_XY
    _crcov = B @ B.T
    if svd_solver == "arnoldi":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        values, vectors = eigsh(_crcov, rank + 3)
    else:
        values, vectors = eigh(_crcov)
    vectors, _vals, _ = eigh_rank_reveal(values, vectors, rank)
    return rsqrt_C_X @ vectors


def fit_rand_reduced_rank_regression(
    C_X: np.ndarray,  # Input covariance matrix
    C_XY: np.ndarray,  # Cross-covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    iterated_power: int,  # Number of power iterations
    rng_seed: Optional[int] = None,  # Random seed
):
    dim = C_X.shape[0]
    reg_input_covariance = C_X + tikhonov_reg * np.identity(dim, dtype=C_X.dtype)
    _crcov = C_XY @ C_XY.T
    rng = np.random.default_rng(rng_seed)
    sketch = rng.standard_normal(
        size=(reg_input_covariance.shape[0], rank + n_oversamples)
    )

    for _ in range(iterated_power):
        _tmp_sketch = solve(reg_input_covariance, sketch, assume_a="pos")
        sketch = _crcov @ _tmp_sketch
        # TODO add power iteration QR normalization.

    sketch_p = solve(reg_input_covariance, sketch, assume_a="pos")

    F_0 = sketch_p.T @ sketch
    F_1 = sketch_p.T @ _crcov @ sketch_p

    values, vectors = eigh(F_1, F_0)
    _norms = weighted_norm(vectors, F_0)
    vectors = vectors @ np.diag(_norms ** (-1.0))
    return sketch_p @ vectors[:, topk(values, rank).indices]


def fit_principal_component_regression(
    C_X: np.ndarray,  # Input covariance matrix
    tikhonov_reg: float = 0.0,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: Optional[int] = None,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. Arnoldi is faster for low ranks.
):
    dim = C_X.shape[0]
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    reg_input_covariance = C_X + tikhonov_reg * np.identity(dim, dtype=C_X.dtype)
    if svd_solver == "arnoldi":
        values, vectors = eigsh(reg_input_covariance, k=rank, which="LM")
    elif svd_solver == "full":
        values, vectors = eigh(reg_input_covariance)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")

    vectors, _, rsqrt_evals = eigh_rank_reveal(values, vectors, rank)
    return vectors @ np.diag(rsqrt_evals)


def fit_rand_principal_component_regression(
    C_X: np.ndarray,  # Input covariance matrix
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    iterated_power: int,  # Number of power iterations
    rcond: float = 2.2e-16,  # Threshold for the singular values
    rng_seed: Optional[int] = None,  # Random seed
):
    dim = C_X.shape[0]
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    reg_input_covariance = C_X + tikhonov_reg * np.identity(dim, dtype=C_X.dtype)

    vectors, values, _ = randomized_svd(
        reg_input_covariance,
        rank,
        n_oversamples=n_oversamples,
        n_iter=iterated_power,
        random_state=rng_seed,
    )

    vectors, _, rsqrt_evals = eigh_rank_reveal(values, vectors, rank, rcond)
    return vectors @ np.diag(rsqrt_evals)


def predict(
    num_steps: int,  # Number of steps to predict (return the last one)
    U: np.ndarray,  # Projection matrix, as returned by the fit functions defined above
    C_XY: np.ndarray,  # Cross-covariance matrix
    phi_Xin: np.ndarray,  # Feature map evaluated on the initial conditions
    phi_X: np.ndarray,  # Feature map evaluated on the training input data
    obs_train_Y: np.ndarray,  # Observable to be predicted evaluated on the output training data
):
    # G = U U.T C_XY
    # G^n = (U)(U.T C_XY U)^(n-1)(U.T C_XY)
    num_train = phi_X.shape[0]
    phi_Xin_dot_U = phi_Xin @ U
    U_C_XY_U = np.linalg.multi_dot([U.T, C_XY, U])
    U_phi_X_obs_Y = np.linalg.multi_dot([U.T, phi_X.T, obs_train_Y]) * (num_train**-1)
    M = np.linalg.matrix_power(U_C_XY_U, num_steps - 1)
    return np.linalg.multi_dot([phi_Xin_dot_U, M, U_phi_X_obs_Y])


def estimator_eig(
    U: np.ndarray,  # Projection matrix, as returned by the fit functions defined above
    C_XY: np.ndarray,  # Cross-covariance matrix
):
    # Using the trick described in https://arxiv.org/abs/1905.11490
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = eig(M, left=True, right=True)

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

    return values, lv, rv


def estimator_modes(
    U: np.ndarray,  # Projection matrix, as returned by the fit functions defined above
    C_XY: np.ndarray,  # Cross-covariance matrix
    phi_X: np.ndarray,  # Feature map evaluated on the training input data
    phi_Xin: np.ndarray,  # Feature map evaluated on the initial conditions
):
    # Using the trick described in https://arxiv.org/abs/1905.11490
    M = np.linalg.multi_dot([U.T, C_XY, U])
    values, lv, rv = eig(M, left=True, right=True)
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
    return (
        rv_in[:, :, None] * lv_obs[:, None, :],
        values,
    )  # [rank, num_init_conditions, num_training_points]


def evaluate_eigenfunction(
    phi_Xin: np.ndarray,  # Feature map evaluated on the initial conditions
    lv_or_rv: np.ndarray,  # Left or right eigenvector, as returned by estimator_eig
):
    return phi_Xin @ lv_or_rv


def svdvals(U, C_XY):
    M = np.linalg.multi_dot([U, U.T, C_XY])
    return np.linalg.svd(M, compute_uv=False)


def estimator_risk(
    cov_Xv: np.ndarray,  # Covariance matrix of the input validation data
    cov_Yv: np.ndarray,  # Covariance matrix of the output validation data
    cov_XYv: np.ndarray,  # Cross-covariance matrix of the validation data
    cov_XY: np.ndarray,  # Cross-covariance matrix of the training data
    U: np.ndarray,  # Projection matrix, as returned by the fit functions defined above
):
    r_Y = np.trace(cov_Yv)
    r_XY = -2 * np.trace(np.linalg.multi_dot([cov_XY.T, U, U.T, cov_XYv]))
    r_X = np.trace(np.linalg.multi_dot([cov_XY.T, U, U.T, cov_Xv, U, U.T, cov_XY]))
    return r_Y + r_XY + r_X
