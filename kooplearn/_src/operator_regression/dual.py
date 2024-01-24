import logging
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eig, eigh, lstsq, qr
from scipy.sparse.linalg import eigs, eigsh
from sklearn.utils.extmath import randomized_svd

from kooplearn._src.linalg import eigh_rank_reveal, weighted_norm
from kooplearn._src.utils import fuzzy_parse_complex, topk

logger = logging.getLogger("kooplearn")


def regularize(M: np.ndarray, reg: float):
    """Regularize a matrix by adding a multiple of the identity matrix to it.
    Args:
        M (np.ndarray): Matrix to regularize.
        reg (float): Regularization parameter.
    Returns:
        np.ndarray: Regularized matrix.
    """
    return M + (M.shape[0] * reg) * np.identity(M.shape[0], dtype=M.dtype)


def add_diagonal(M: np.ndarray, alpha: float):
    np.fill_diagonal(M, M.diagonal() + alpha)


# Reduced Rank Algorithms
def _filter_reduced_rank_svals(svals_sq, vecs, rank):
    rcond = 10.0 * vecs.shape[0] * np.finfo(vecs.dtype).eps
    # Filtering procedure.
    # First: Sort singular values by magnitude
    sort_perm = topk(svals_sq.real, len(svals_sq)).indices
    svals_sq = svals_sq[sort_perm]
    vecs = vecs[:, sort_perm]

    # Second: create a mask which is True when the real part of the eigenvalue is negative or the imaginary part is nonzero
    is_invalid = np.logical_or(
        np.abs(np.real(svals_sq)) <= rcond, np.imag(svals_sq) != 0
    )
    # Third, check if any is invalid take the first occurrence of a True value in the mask and filter everything after that
    if np.any(is_invalid):
        first_invalid = np.argmax(
            is_invalid
        )  # In the case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        svals_filtered = svals_sq[: min(first_invalid, rank)]
        vecs_filtered = vecs[:, : min(first_invalid, rank)]
        if first_invalid < rank:
            # Logging. Print the number of discarded eigenvalues, as well as the magnitude of the largest one.
            logger.warning(
                f"Warning: Discarted {rank - len(svals_filtered)} dimensions of the {rank} requested, consider decreasing the rank. The smallest squared singular value kept is: {np.min(np.abs(svals_sq[:first_invalid])):.3e}."
            )
    else:
        svals_filtered = svals_sq[:rank]
        vecs_filtered = vecs[:, :rank]

    # Fourth assert that the eigenvectors do not have any imaginary part
    assert np.all(
        np.imag(vecs_filtered) == 0
    ), "The eigenvectors should be real. Decrease the rank or increase the regularization strength."

    # Fifth: take the real part of the eigenvectors
    vecs_filtered = np.real(vecs_filtered)
    svals_filtered = np.real(svals_filtered)
    return svals_filtered, vecs_filtered


def fit_reduced_rank_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    kernel_Y: np.ndarray,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tikhonov_reg == 0.0:
        return _fit_reduced_rank_regression_noreg(
            kernel_X, kernel_Y, rank, svd_solver=svd_solver
        )
    else:
        n_pts = kernel_X.shape[0]
        rcond = 10.0 * n_pts * np.finfo(kernel_X.dtype).eps
        penalty = max(rcond, tikhonov_reg) * n_pts
        A = np.multiply(kernel_Y, n_pts ** (-0.5)) @ np.multiply(
            kernel_X, n_pts ** (-0.5)
        )

        add_diagonal(kernel_X, penalty)
        # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
        # Prefer svd_solver == 'randomized' in such a case.
        if svd_solver == "arnoldi":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            _num_arnoldi_eigs = min(rank + 3, A.shape[0])
            sigma_sq, vecs = eigs(A, k=_num_arnoldi_eigs, M=kernel_X)
        else:  # 'full'
            sigma_sq, vecs = eig(A, kernel_X, overwrite_a=True, overwrite_b=True)
        add_diagonal(kernel_X, -penalty)

        svals_filtered, vecs_filtered = _filter_reduced_rank_svals(sigma_sq, vecs, rank)
        # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
        if not np.all(np.abs(svals_filtered) >= tikhonov_reg):
            logger.warning(
                f"Warning: {(np.abs(svals_filtered) < tikhonov_reg).sum()} out of the {len(svals_filtered)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
            )
        # Eigenvector normalization
        kernel_X_vecs = np.dot(np.multiply(kernel_X, n_pts ** (-0.5)), vecs_filtered)
        vecs_norms = np.sum(
            kernel_X_vecs**2
            + tikhonov_reg * kernel_X_vecs * vecs_filtered * (n_pts**0.5),
            axis=0,
        ) ** (0.5)
        U = vecs_filtered / vecs_norms
        V = kernel_X @ U
        svals = np.flip(np.sort(np.abs(sigma_sq))) ** 0.5
        return U, V, svals


def _fit_reduced_rank_regression_noreg(
    K_X: np.ndarray,  # Kernel matrix of the input data
    K_Y: np.ndarray,  # Kernel matrix of the output data
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Solve the Hermitian eigenvalue problem to find V
    logger.warning(
        "The least-squares solution (tikhonov_reg == 0) of the reduced rank problem in the kernel setting is computationally very inefficient. Consider adding a small regularization parameter."
    )

    values_X, U_X = eigh(K_X)
    U_X, _, _ = eigh_rank_reveal(values_X, U_X, K_X.shape[0], verbose=False)
    proj_X = U_X @ U_X.T
    L = proj_X @ K_Y
    if svd_solver != "full":
        sigma_sq, vecs = eigs(L, rank + 3)
    else:
        sigma_sq, vecs = eig(L, overwrite_a=True)
    svals_filtered, V = _filter_reduced_rank_svals(sigma_sq, vecs, rank)
    # Normalize V
    _V_norm = np.linalg.norm(V, ord=2, axis=0) / np.sqrt(V.shape[0])
    rcond = 10.0 * K_X.shape[0] * np.finfo(K_X.dtype).eps
    _inv_V_norm = np.where(_V_norm < rcond, 0.0, _V_norm**-1)
    V = V / _inv_V_norm
    # Solve the least squares problem to determine U
    U = lstsq(K_X, V)[0]
    svals = np.flip(np.sort(np.abs(sigma_sq))) ** 0.5

    return U, V, svals


def fit_nystroem_reduced_rank_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input inducing points
    kernel_Y: np.ndarray,  # Kernel matrix of the output inducing points
    kernel_Xnys: np.ndarray,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: np.ndarray,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float = 0.0,  # Tikhonov regularization parameter
    rank: Optional[int] = None,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = kernel_Xnys.shape[0]
    num_centers = kernel_X.shape[0]

    eps = kernel_X.shape[0] * np.finfo(kernel_X.dtype).eps
    reg = max(eps, tikhonov_reg)

    # LHS of the generalized eigenvalue problem
    kernel_YX_nys = (kernel_Ynys.T @ kernel_Xnys) * (dim**-0.5)

    _tmp_YX = lstsq(kernel_Y, kernel_YX_nys)[0]
    kernel_XYX = kernel_YX_nys.T @ _tmp_YX
    kernel_XYX = (kernel_XYX + kernel_XYX.T) * 0.5  # Symmetrize for numerical stability

    # RHS of the generalized eigenvalue problem
    kernel_Xnys_sq = kernel_Xnys.T @ kernel_Xnys + reg * dim * kernel_X

    A = lstsq(kernel_Xnys_sq, kernel_XYX)[0]
    if svd_solver == "full":
        values, vectors = eig(A)  # normalization leads to needing to invert evals
    elif svd_solver == "arnoldi":
        _num_arnoldi_eigs = min(rank + 3, kernel_X.shape[0])
        values, vectors = eigs(A, k=_num_arnoldi_eigs)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")

    # Remove elements in the kernel of kernel_XYX
    _threshold = 10.0 * np.finfo(vectors.dtype).eps * vectors.shape[0]
    relative_norm_sq = (
        (num_centers**-1)
        * np.sum(vectors * (kernel_XYX @ vectors), axis=0)
        / (np.linalg.norm(vectors, axis=0) ** 2)
    )
    is_in_kernel = np.abs(relative_norm_sq) < _threshold
    values = values[~is_in_kernel]
    vectors = vectors[:, ~is_in_kernel]

    values_filtered, vectors = _filter_reduced_rank_svals(values, vectors, rank)
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values_filtered) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values_filtered) < tikhonov_reg).sum()} out of the {len(values_filtered)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )
    # Eigenvector normalization
    normalization_csts = (np.sum(vectors * (kernel_XYX @ vectors), axis=0)) ** 0.5
    vectors = (num_centers**0.5) * vectors / normalization_csts
    U = A @ vectors
    V = _tmp_YX @ vectors * (dim**0.5)

    svals = np.flip(np.sort(np.abs(values))) ** 0.5
    return U.real, V.real, svals


def fit_rand_reduced_rank_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    kernel_Y: np.ndarray,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    optimal_sketching: bool,  # Whether to use optimal sketching (slower but more accurate) or not.
    iterated_power: int,  # Number of iterations of the power method
    rng_seed: Optional[
        int
    ] = None,  # Seed for the random number generator (for reproducibility)
    precomputed_cholesky=None,  # Precomputed Cholesky decomposition. Should be the output of cho_factor evaluated on the regularized kernel matrix.
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    dim = kernel_X.shape[0]
    inv_dim = dim ** (-1.0)

    penalty = dim * tikhonov_reg
    add_diagonal(kernel_X, penalty)
    if precomputed_cholesky is None:
        cholesky_decomposition = cho_factor(kernel_X)
    else:
        cholesky_decomposition = precomputed_cholesky
    add_diagonal(kernel_X, -penalty)

    sketch_dimension = rank + n_oversamples

    if optimal_sketching:
        Cov = inv_dim * kernel_Y
        sketch = rng.multivariate_normal(
            np.zeros(dim, dtype=kernel_Y.dtype), Cov, size=sketch_dimension
        ).T
    else:
        sketch = rng.standard_normal(size=(dim, sketch_dimension))

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        sketch = (inv_dim * kernel_Y) @ (
            sketch - penalty * cho_solve(cholesky_decomposition, sketch)
        )
        sketch, _ = qr(sketch, mode="economic")  # QR re-orthogonalization

    kernel_X_sketch = cho_solve(cholesky_decomposition, sketch)
    _M = sketch - penalty * kernel_X_sketch

    F_0 = sketch.T @ sketch - penalty * (sketch.T @ kernel_X_sketch)  # Symmetric
    F_0 = 0.5 * (F_0 + F_0.T)
    F_1 = _M.T @ (inv_dim * (kernel_Y @ _M))

    values, vectors = eig(lstsq(F_0, F_1)[0])

    # Remove elements in the kernel of F_0
    _threshold = 10.0 * np.finfo(vectors.dtype).eps * vectors.shape[0]
    relative_norm_sq = (
        np.sum(vectors.conj() * (F_0 @ vectors), axis=0)
        / np.linalg.norm(vectors, axis=0) ** 2
    )
    is_in_kernel = np.abs(relative_norm_sq) < _threshold
    values = values[~is_in_kernel]
    vectors = vectors[:, ~is_in_kernel]

    _, vectors = _filter_reduced_rank_svals(values, vectors, rank)

    normalization_csts = (np.sum(vectors * (F_0 @ vectors), axis=0)) ** 0.5
    vectors = vectors / normalization_csts

    U = (dim**0.5) * np.asfortranarray(kernel_X_sketch @ vectors)
    V = (dim**0.5) * np.asfortranarray(_M @ vectors)

    svals = np.flip(np.sort(np.abs(values))) ** 0.5
    return U.real, V.real, svals


# Principal Component Algorithms
def fit_principal_component_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    tikhonov_reg: float = 0.0,  # Tikhonov regularization parameter, can be zero
    rank: Optional[int] = None,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
) -> tuple[np.ndarray, np.ndarray]:
    dim = kernel_X.shape[0]
    add_diagonal(kernel_X, dim * tikhonov_reg)
    if svd_solver == "arnoldi":
        _num_arnoldi_eigs = min(rank + 3, kernel_X.shape[0])
        values, vectors = eigsh(kernel_X, k=_num_arnoldi_eigs)
    elif svd_solver == "full":
        values, vectors = eigh(kernel_X)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal(kernel_X, -dim * tikhonov_reg)

    filtered_vectors, filtered_values, rsqrt_values = eigh_rank_reveal(
        values, vectors, rank
    )
    Q = np.sqrt(dim) * filtered_vectors * (rsqrt_values)
    kernel_X_eigvalsh = (np.flip(np.sort(np.abs(values))) / dim) ** 0.5
    return Q, Q, kernel_X_eigvalsh


def fit_nystroem_principal_component_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input inducing points
    kernel_Y: np.ndarray,  # Kernel matrix of the output inducing points
    kernel_Xnys: np.ndarray,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: np.ndarray,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float = 0.0,  # Tikhonov regularization parameter (can be 0)
    rank: Optional[int] = None,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = kernel_X.shape[0]
    eps = kernel_X.shape[0] * np.finfo(kernel_X.dtype).eps
    reg = max(eps, tikhonov_reg)
    kernel_Xnys_sq = kernel_Xnys.T @ kernel_Xnys
    add_diagonal(kernel_X, reg * dim)
    if svd_solver == "full":
        values, vectors = eigh(
            kernel_Xnys_sq, kernel_X
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arnoldi":
        _num_arnoldi_eigs = min(rank + 3, kernel_X.shape[0])
        values, vectors = eigsh(
            kernel_Xnys_sq,
            M=kernel_X,
            k=_num_arnoldi_eigs,
            which="LM",
        )
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal(kernel_X, -reg * dim)
    vectors, values, rsqrt_values = eigh_rank_reveal(values, vectors, rank)

    U = np.sqrt(dim) * vectors * (rsqrt_values)
    V = np.linalg.multi_dot([kernel_Ynys.T, kernel_Xnys, vectors])
    add_diagonal(kernel_Y, eps * dim)
    V = lstsq(kernel_Y, V)[0]
    add_diagonal(kernel_Y, -eps * dim)
    V = np.sqrt(dim) * V * (rsqrt_values)

    kernel_X_eigvalsh = (np.flip(np.sort(np.abs(values))) / kernel_Xnys.shape[0]) ** 0.5
    return U, V, kernel_X_eigvalsh


def fit_rand_principal_component_regression(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    tikhonov_reg: float,  # Tikhonov regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    iterated_power: int,  # Number of iterations for the power method
    rng_seed: Optional[int] = None,  # Seed for the random number generator
):
    dim = kernel_X.shape[0]
    add_diagonal(kernel_X, dim * tikhonov_reg)
    vectors, values, _ = randomized_svd(
        kernel_X,
        rank,
        n_oversamples=n_oversamples,
        n_iter=iterated_power,
        random_state=rng_seed,
    )
    add_diagonal(kernel_X, -dim * tikhonov_reg)

    filtered_vectors, filtered_values, rsqrt_values = eigh_rank_reveal(
        values, vectors, rank
    )
    Q = np.sqrt(dim) * filtered_vectors * (rsqrt_values)
    kernel_X_eigvalsh = (np.flip(np.sort(np.abs(values))) / dim) ** 0.5
    return Q, Q, kernel_X_eigvalsh


def predict(
    num_steps: int,  # Number of steps to predict (return the last one)
    U: np.ndarray,  # Projection matrix: first output of the fit functions defined above
    V: np.ndarray,  # Projection matrix: second output of the fit functions defined above
    K_YX: np.ndarray,  # Kernel matrix between the output data and the input data
    K_Xin_X: np.ndarray,  # Kernel matrix between the initial conditions and the input data
    obs_train_Y: np.ndarray,  # Observable to be predicted evaluated on the output training data
) -> np.ndarray:
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    dim = U.shape[0]
    rsqrt_dim = dim ** (-0.5)
    K_dot_U = rsqrt_dim * K_Xin_X @ U
    V_dot_obs = rsqrt_dim * V.T @ obs_train_Y
    V_K_XY_U = (dim**-1) * np.linalg.multi_dot([V.T, K_YX, U])
    M = np.linalg.matrix_power(V_K_XY_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])


def estimator_eig(
    U: np.ndarray,  # Projection matrix: first output of the fit functions defined above
    V: np.ndarray,  # Projection matrix: second output of the fit functions defined above
    K_X: np.ndarray,  # Kernel matrix of the input data
    K_YX: np.ndarray,  # Kernel matrix between the output data and the input data
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # SUV.TZ -> V.T K_YX U (right ev = SUvr, left ev = ZVvl)
    r_dim = (K_X.shape[0]) ** (-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim * K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim * K_X, U])

    values, vl, vr = eig(W_YX, left=True, right=True)  # Left -> V, Right -> U
    values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    rcond = 10.0 * U.shape[0] * np.finfo(U.dtype).eps
    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    r_normr = np.where(norm_r < rcond, 0.0, norm_r**-1)
    vr = vr * r_normr

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    r_norm_l = np.where(np.abs(norm_l) < rcond, 0.0, norm_l**-1)
    vl = vl * r_norm_l
    return values, V @ vl, U @ vr


def estimator_modes(K_Xin_X: np.ndarray, rv: np.ndarray, lv: np.ndarray):
    r_dim = lv.shape[0] ** -0.5
    rv_in = evaluate_eigenfunction(K_Xin_X, rv).T  # [rank, num_initial_conditions]
    lv_obs = r_dim * lv.T  # [rank, num_observations]
    return (
        rv_in[:, :, None] * lv_obs[:, None, :]
    )  # [rank, num_init_conditions, num_training_points]


def evaluate_eigenfunction(
    K_Xin_X_or_Y: np.ndarray,
    # Kernel matrix between the initial conditions and the input data (right eigenfunctions) or the output data
    # (left eigenfunctions)
    vr_or_vl: np.ndarray,  # Right eigenvectors or left eigenvectors, as returned by the estimator_eig function
):
    rsqrt_dim = (K_Xin_X_or_Y.shape[1]) ** (-0.5)
    return np.linalg.multi_dot([rsqrt_dim * K_Xin_X_or_Y, vr_or_vl])


def svdvals(
    U: np.ndarray,  # Projection matrix: first output of the fit functions defined above
    V: np.ndarray,  # Projection matrix: second output of the fit functions defined above
    K_X: np.ndarray,  # Kernel matrix of the input data
    K_Y: np.ndarray,  # Kernel matrix of the output data
):
    # Inefficient implementation
    rdim = (K_X.shape[0]) ** (-1)
    A = np.linalg.multi_dot([V.T, rdim * K_Y, V])
    B = np.linalg.multi_dot([U.T, rdim * K_X, U])
    v = eig(A @ B, left=False, right=False)
    # Clip the negative values
    v = v.real
    v[v < 0] = 0
    return np.sqrt(v)


def estimator_risk(
    kernel_Yv: np.ndarray,  # Kernel matrix of the output validation data
    kernel_Y: np.ndarray,  # Kernel matrix of the output training data
    kernel_XXv: np.ndarray,  # Cross-Kernel matrix of the input train/validation data
    kernel_YYv: np.ndarray,  # Cross-Kernel matrix of the output train/validation data
    U: np.ndarray,  # Projection matrix: first output of the fit functions defined above
    V: np.ndarray,  # Projection matrix: second output of the fit functions defined above
):
    rdim_train = (kernel_Y.shape[0]) ** (-1)
    rdim_val = (kernel_Yv.shape[0]) ** (-1)

    r_Y = rdim_val * np.trace(kernel_Yv)
    r_XY = (
        -2
        * rdim_val
        * rdim_train
        * np.trace(np.linalg.multi_dot([kernel_YYv.T, V, U.T, kernel_XXv]))
    )
    r_X = (
        rdim_val
        * (rdim_train**2)
        * np.trace(
            np.linalg.multi_dot([kernel_XXv.T, U, V.T, kernel_Y, V, U.T, kernel_XXv])
        )
    )
    return r_Y + r_XY + r_X
