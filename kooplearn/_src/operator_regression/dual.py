from typing import Optional
import logging
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig, eigh, LinAlgError, pinvh, lstsq
from scipy.sparse.linalg import eigs, eigsh, lsqr
from scipy.sparse.linalg._eigen.arpack.arpack import IterInv
from sklearn.utils.extmath import randomized_svd
from kooplearn._src.utils import topk, modified_QR, weighted_norm


def regularize(M: ArrayLike, reg: float):
    """Regularize a matrix by adding a multiple of the identity matrix to it.
    Args:
        M (ArrayLike): Matrix to regularize.
        reg (float): Regularization parameter.
    Returns:
        ArrayLike: Regularized matrix.
    """
    return M + reg * M.shape[0] * np.identity(M.shape[0], dtype=M.dtype)


def fit_reduced_rank_regression_tikhonov(
        K_X: ArrayLike,  # Kernel matrix of the input data
        K_Y: ArrayLike,  # Kernel matrix of the output data
        tikhonov_reg: float,  # Tikhonov regularization parameter, can be 0
        rank: int,  # Rank of the estimator
        svd_solver: str = 'arnoldi',  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
) -> tuple[ArrayLike, ArrayLike] or tuple[ArrayLike, ArrayLike, ArrayLike]:
    if tikhonov_reg == 0.:
        return _fit_reduced_rank_regression_noreg(K_X, K_Y, rank, svd_solver=svd_solver,
                                                  _return_singular_values=_return_singular_values)
    else:
        dim = K_X.shape[0]
        rsqrt_dim = dim ** (-0.5)
        # Rescaled Kernel matrices
        K_Xn = K_X * rsqrt_dim
        K_Yn = K_Y * rsqrt_dim

        K = K_Yn @ K_Xn
        # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
        # Prefer svd_solver == 'randomized' in such a case.
        if svd_solver == 'arnoldi':
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            sigma_sq, U = eigs(K, rank + 3, regularize(K_X, tikhonov_reg))
        else:  # 'full'
            sigma_sq, U = eig(K, regularize(K_X, tikhonov_reg))

        max_imag_part = np.max(U.imag)
        if max_imag_part >= 2.2e-10:
            logging.warn(f"The computed projector is not real. The Kernel matrix is severely ill-conditioned.")
        U = np.real(U)

        # Post-process U. Promote numerical stability via additional QR decoposition if necessary.
        U = U[:, topk(sigma_sq.real, rank).indices]

        norm_inducing_op = (K_Xn @ (K_Xn.T)) + tikhonov_reg * K_X
        U, _, columns_permutation = modified_QR(U, M=norm_inducing_op, column_pivoting=True)
        U = U[:, np.argsort(columns_permutation)]
        if U.shape[1] < rank:
            logging.warn(
                f"The numerical rank of the projector is smaller than the selected rank ({rank}). {rank - U.shape[1]} "
                f"degrees of freedom will be ignored.")
            _zeroes = np.zeros((U.shape[0], rank - U.shape[1]))
            U = np.c_[U, _zeroes]
            assert U.shape[1] == rank
        V = K_X @ np.asfortranarray(U)
        if _return_singular_values:
            return U, V, sigma_sq
        else:
            return U, V


def _fit_reduced_rank_regression_noreg(
        K_X: ArrayLike,  # Kernel matrix of the input data
        K_Y: ArrayLike,  # Kernel matrix of the output data
        rank: int,  # Rank of the estimator
        svd_solver: str = 'arnoldi',  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
) -> tuple[ArrayLike, ArrayLike] or tuple[ArrayLike, ArrayLike, ArrayLike]:
    # Solve the Hermitian eigenvalue problem to find V
    if svd_solver != 'full':
        sigma_sq, V = eigsh(K_Y, rank)
    else:
        sigma_sq, V = eigh(K_Y)
        V = V[:, topk(sigma_sq, rank).indices]

    # Normalize V
    _V_norm = np.linalg.norm(V, ord=2, axis=0) / np.sqrt(V.shape[0])
    V = V @ np.diag(_V_norm ** -1)

    # Solve the least squares problem to determine U
    if svd_solver != 'full':
        U = np.zeros_like(V)
        for i in range(U.shape[1]):
            U[:, i] = lsqr(K_X, V[:, i])[0]  # Not optimal with this explicit loop
    else:
        U = lstsq(K_X, V)[0]
    if _return_singular_values:
        return U, V, sigma_sq
    else:
        return U, V


def fit_nystrom_reduced_rank_regression_tikhonov(
        N_X: ArrayLike,  # Kernel matrix of the input inducing points
        N_Y: ArrayLike,  # Kernel matrix of the output inducing points
        KN_X: ArrayLike,  # Kernel matrix between the input data and the input inducing points
        KN_Y: ArrayLike,  # Kernel matrix between the output data and the output inducing points
        tikhonov_reg: float,  # Tikhonov regularization parameter
        rank: int,  # Rank of the estimator
) -> tuple[ArrayLike, ArrayLike]:
    num_training_pts = KN_X.shape[0]
    NKy_KNx = (KN_Y.T) @ KN_X
    _B = lstsq(N_Y, NKy_KNx)[0]
    NKN = (NKy_KNx.T) @ _B
    G = (KN_X.T) @ KN_X + tikhonov_reg * num_training_pts * N_X
    S, W = eigh(NKN, G)
    # Low-rank projection
    W = W[:, topk(S, rank).indices]
    # Normalize the eigenvectors
    W = W @ np.diag(weighted_norm(W, NKN) ** -1)
    V = _B @ W
    U = NKN @ W
    U = lstsq(G, U)[0]
    return U, V


def fit_rand_reduced_rank_regression_tikhonov(
        K_X: ArrayLike,  # Kernel matrix of the input data
        K_Y: ArrayLike,  # Kernel matrix of the output data
        tikhonov_reg: float,  # Tikhonov regularization parameter
        rank: int,  # Rank of the estimator
        n_oversamples: int,  # Number of oversamples
        optimal_sketching: bool,  # Whether to use optimal sketching (slower but more accurate) or not.
        iterated_power: int,  # Number of iterations of the power method
        rng_seed: Optional[int] = None,  # Seed for the random number generator (for reproducibility)
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
) -> tuple[ArrayLike, ArrayLike] or tuple[ArrayLike, ArrayLike, ArrayLike]:
    dim = K_X.shape[0]
    inv_dim = dim ** (-1.0)
    alpha = dim * tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    K_reg_inv = IterInv(K_X + tikhonov)
    l = rank + n_oversamples
    rng = np.random.default_rng(rng_seed)
    if optimal_sketching:
        Cov = inv_dim * K_Y
        Om = rng.multivariate_normal(np.zeros(dim, dtype=K_X.dtype), Cov, size=l).T
    else:
        Om = rng.standard_normal(size=(dim, l))

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        Om = (inv_dim * K_Y) @ (Om - alpha * K_reg_inv @ Om)
    KOm = K_reg_inv @ Om
    KOmp = Om - alpha * KOm

    F_0 = (Om.T @ KOmp)
    F_1 = (KOmp.T @ (inv_dim * (K_Y @ KOmp)))

    # Generation of matrices U and V.
    try:
        sigma_sq, Q = eigh(F_1, F_0)
    except LinAlgError:
        sigma_sq, Q = eig(pinvh(F_0) @ F_1)

    Q_norm = np.sum(Q.conj() * (F_0 @ Q), axis=0)
    Q = Q @ np.diag(Q_norm ** -0.5)
    _idxs = topk(sigma_sq.real, rank).indices
    sigma_sq = sigma_sq.real

    Q = Q[:, _idxs]
    U = (dim ** 0.5) * np.asfortranarray(KOm @ Q)
    V = (dim ** 0.5) * np.asfortranarray(KOmp @ Q)
    if _return_singular_values:
        return U.real, V.real, sigma_sq
    else:
        return U.real, V.real


def fit_tikhonov(
        K_X: ArrayLike,  # Kernel matrix of the input data
        tikhonov_reg: float = 0.0,  # Tikhonov regularization parameter, can be zero
        rank: Optional[int] = None,  # Rank of the estimator
        svd_solver: str = 'arnoldi',  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
        rcond: float = 2.2e-16  # Threshold for the singular values
) -> tuple[ArrayLike, ArrayLike]:
    dim = K_X.shape[0]
    if rank is None:
        rank = dim
    assert rank <= dim, f"Rank too high. The maximum value for this problem is {dim}"
    alpha = dim * tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    if svd_solver == 'arnoldi':
        S, V = eigsh(K_X + tikhonov, k=rank)
    elif svd_solver == 'full':
        S, V = eigh(K_X + tikhonov)
    S, V = _postprocess_tikhonov_fit(S, V, rank, dim, rcond)
    return V, V


def fit_nystrom_tikhonov(
        N_X: ArrayLike,  # Kernel matrix of the input inducing points
        N_Y: ArrayLike,  # Kernel matrix of the output inducing points
        KN_X: ArrayLike,  # Kernel matrix between the input data and the input inducing points
        KN_Y: ArrayLike,  # Kernel matrix between the output data and the output inducing points
        tikhonov_reg: float = 0.0,  # Tikhonov regularization parameter (can be 0)
        rank: Optional[int] = None,  # Rank of the estimator
        rcond: float = 2.2e-16,  # Threshold for the singular values
) -> tuple[ArrayLike, ArrayLike]:
    # Not using the Rank parameter fix it.
    num_training_pts = KN_X.shape[0]
    NKy_KNx = (KN_Y.T) @ KN_X
    G = (KN_X.T) @ KN_X + tikhonov_reg * num_training_pts * N_X
    U = lstsq(G, NKy_KNx)[0]
    V = lstsq(N_Y, U)[0]
    return U, V


def fit_rand_tikhonov(
        K_X: ArrayLike,  # Kernel matrix of the input data
        tikhonov_reg: float,  # Tikhonov regularization parameter
        rank: int,  # Rank of the estimator
        n_oversamples: int,  # Number of oversamples
        iterated_power: int,  # Number of iterations for the power method
        rcond: float = 2.2e-16,  # Threshold for the singular values
        rng_seed: Optional[int] = None,  # Seed for the random number generator
        _return_singular_values: bool = False
        # Whether to return the singular values of the projector. (Development purposes)
):
    dim = K_X.shape[0]
    alpha = dim * tikhonov_reg
    tikhonov = np.identity(dim, dtype=K_X.dtype) * alpha
    V, S, _ = randomized_svd(K_X + tikhonov, rank, n_oversamples=n_oversamples, n_iter=iterated_power,
                             random_state=rng_seed)
    S, V = _postprocess_tikhonov_fit(S, V, rank, dim, rcond)
    if _return_singular_values:
        return V, V, S
    else:
        return V, V


def _postprocess_tikhonov_fit(
        S: ArrayLike,  # Singular values
        V: ArrayLike,  # Eigenvectors
        rank: int,  # Desired rank
        dim: int,  # Dimension of the problem
        rcond: float  # Threshold for the singular values
):
    top_svals = topk(S, rank)
    V = V[:, top_svals.indices]
    S = top_svals.values

    _test = S > rcond
    if all(_test):
        V = np.sqrt(dim) * (V @ np.diag(S ** -0.5))
    else:
        V = np.sqrt(dim) * (V[:, _test] @ np.diag(S[_test] ** -0.5))
        logging.warn(
            f"The numerical rank of the projector ({V.shape[1]}) is smaller than the selected rank ({rank}). {rank - V.shape[1]} degrees of freedom will be ignored.")
        _zeroes = np.zeros((V.shape[0], rank - V.shape[1]))
        V = np.c_[V, _zeroes]
        assert V.shape[1] == rank
    return S, V


def predict(
        num_steps: int,  # Number of steps to predict (return the last one)
        U: ArrayLike,  # Projection matrix: first output of the fit functions defined above
        V: ArrayLike,  # Projection matrix: second output of the fit functions defined above
        K_YX: ArrayLike,  # Kernel matrix between the output data and the input data
        K_Xin_X: ArrayLike,  # Kernel matrix between the initial conditions and the input data
        obs_train_Y: ArrayLike  # Observable to be predicted evaluated on the output training data
) -> ArrayLike:
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    dim = U.shape[0]
    rsqrt_dim = dim ** (-0.5)
    K_dot_U = rsqrt_dim * K_Xin_X @ U
    V_dot_obs = rsqrt_dim * (V.T) @ obs_train_Y
    V_K_XY_U = (dim ** -1) * np.linalg.multi_dot([V.T, K_YX, U])
    M = np.linalg.matrix_power(V_K_XY_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])


def estimator_eig(
        U: ArrayLike,  # Projection matrix: first output of the fit functions defined above
        V: ArrayLike,  # Projection matrix: second output of the fit functions defined above
        K_X: ArrayLike,  # Kernel matrix of the input data
        K_Y: ArrayLike,  # Kernel matrix of the output data
        K_YX: ArrayLike  # Kernel matrix between the output data and the input data
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    # SUV.TZ -> V.T K_YX U (right ev = SUvr, left ev = ZVvl)
    r_dim = (K_X.shape[0]) ** (-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim * K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim * K_X, U])

    values, vl, vr = eig(W_YX, left=True, right=True)  # Left -> V, Right -> U

    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    vr = vr @ np.diag(norm_r ** (-1))

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    vl = vl / norm_l
    return values, vl, vr


def evaluate_eigenfunction(
        K_Xin_X_or_Y: ArrayLike,
        # Kernel matrix between the initial conditions and the input data (right eigenfunctions) or the output data
        # (left eigenfunctions)
        U_or_V: ArrayLike,
        # Projection matrix: first output of the fit functions defined above (right eigenfunctions) or second output
        # (left eigenfunctions)
        vr_or_vl: ArrayLike  # Right eigenvectors or left eigenvectors, as returned by the estimator_eig function
):
    rsqrt_dim = (U_or_V.shape[0]) ** (-0.5)
    return np.linalg.multi_dot([rsqrt_dim * K_Xin_X_or_Y, U_or_V, vr_or_vl])


def svdvals(
        U: ArrayLike,  # Projection matrix: first output of the fit functions defined above
        V: ArrayLike,  # Projection matrix: second output of the fit functions defined above
        K_X: ArrayLike,  # Kernel matrix of the input data
        K_Y: ArrayLike,  # Kernel matrix of the output data
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
