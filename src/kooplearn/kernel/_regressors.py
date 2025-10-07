"""Kernel-based regressors for linear operators."""


from math import sqrt
from typing import Literal
from warnings import warn

import numpy as np
import scipy.linalg
from numpy import ndarray
from scipy.sparse.linalg import eigs, eigsh
from sklearn.utils.extmath import randomized_svd

from kooplearn.kernel.linalg import add_diagonal_, stable_topk, weighted_norm
from kooplearn.kernel.structs import EigResult, FitResult
from kooplearn.kernel.utils import sanitize_complex_conjugates

__all__ = [
    "eig",
    "evaluate_eigenfunction",
    "nystroem_pcr",
    "nystroem_reduced_rank",
    "pcr",
    "predict",
    "rand_reduced_rank",
    "reduced_rank",
]


def estimator_risk(
    fit_result: FitResult,
    kernel_Yv: np.ndarray,
    kernel_Y: np.ndarray,
    kernel_XXv: np.ndarray,
    kernel_YYv: np.ndarray,
    ) -> float:
    """
    Estimate the validation risk of a fitted kernel regressor.

    Parameters
    ----------
    fit_result : FitResult
        Fitted model containing matrices ``U`` and ``V``.
    kernel_Yv : ndarray of shape (N_val, N_val)
        Kernel matrix of the output validation data.
    kernel_Y : ndarray of shape (N_train, N_train)
        Kernel matrix of the output training data.
    kernel_XXv : ndarray of shape (N_train, N_val)
        Cross-kernel matrix between input train and validation data.
    kernel_YYv : ndarray of shape (N_train, N_val)
        Cross-kernel matrix between output train and validation data.

    Returns
    -------
    risk : float
        Estimated validation risk.
    """
    U = fit_result["U"]
    V = fit_result["V"]
    rdim_train = kernel_Y.shape[0] ** (-1)
    rdim_val = kernel_Yv.shape[0] ** (-1)

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


def eig(
        fit_result: FitResult,
        K_X: ndarray, 
        K_YX: ndarray
        ) -> EigResult:
    """
    Compute the eigendecomposition of a fitted kernel regressor.

    Parameters
    ----------
    fit_result : FitResult
        Fit result containing matrices ``U`` and ``V``.
    K_X : ndarray of shape (N, N)
        Kernel matrix of the input data.
    K_YX : ndarray of shape (N, N)
        Cross-kernel matrix between the output and input data.

    Returns
    -------
    EigResult
        Dictionary containing:
        - ``values`` : Eigenvalues of the operator.
        - ``left`` : Left eigenfunctions (N, R).
        - ``right`` : Right eigenfunctions (N, R).
    """
    # SUV.TZ -> V.T K_YX U (right ev = SUvr, left ev = ZVvl)
    U = fit_result["U"]
    V = fit_result["V"]
    r_dim = K_X.shape[0] ** (-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim * K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim * K_X, U])

    values, vl, vr = scipy.linalg.eig(W_YX, left=True, right=True)  # Left -> V, Right -> U
    values = sanitize_complex_conjugates(values)
    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    rcond = 1000.0 * np.finfo(U.dtype).eps
    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    norm_r = np.where(norm_r < rcond, np.inf, norm_r)
    vr = vr / norm_r

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    norm_l = np.where(np.abs(norm_l) < rcond, np.inf, norm_l)
    vl = vl / norm_l
    result: EigResult = {"values": values, "left": V @ vl, "right": U @ vr}
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    K_Xin_X_or_Y: ndarray,
):
    """
    Evaluate left or right eigenfunctions on new data.

    Parameters
    ----------
    eig_result : EigResult
        Dictionary containing eigenvalues and eigenfunctions.
    which : {'left', 'right'}
        Which set of eigenfunctions to evaluate.
    K_Xin_X_or_Y : ndarray of shape (N_eval, N_train)
        Kernel matrix between evaluation points and training data.

    Returns
    -------
    ndarray of shape (N_eval, R)
        Evaluated eigenfunctions.
    """
    vr_or_vl = eig_result[which]
    rsqrt_dim = (K_Xin_X_or_Y.shape[1]) ** (-0.5)
    return np.linalg.multi_dot([rsqrt_dim * K_Xin_X_or_Y, vr_or_vl])


def estimator_modes(
            eig_result: EigResult,
            K_Xin_X: np.ndarray):
    """
    Compute dynamic modes associated with eigenfunctions.

    Parameters
    ----------
    eig_result : EigResult
        Dictionary containing left and right eigenfunctions.
    K_Xin_X : ndarray of shape (N_eval, N_train)
        Kernel matrix between evaluation and training data.

    Returns
    -------
    ndarray of shape (R, N_eval, N_train)
        Outer product of left and right eigenfunctions (dynamic modes).
    """
    lv = eig_result["left"]
    r_dim = lv.shape[0] ** -0.5
    rv_in = evaluate_eigenfunction(
        eig_result, 'right', K_Xin_X
        ).T  # [rank, num_initial_conditions]
    lv_obs = r_dim * lv.T  # [rank, num_observations]
    return (
        rv_in[:, :, None] * lv_obs[:, None, :]
    )  # [rank, num_init_conditions, num_training_points]


def predict(
    num_steps: int,
    fit_result: FitResult,
    kernel_YX: ndarray,
    kernel_Xin_X: ndarray,
    obs_train_Y: ndarray,
) -> ndarray:
    """
    Predicts future states given initial values using a fitted regressor.

    Parameters
    ----------
    num_steps : int
        Number of prediction steps forward (returns the last
        prediction).
    fit_result : FitResult
        Dictionary containing matrices ``U`` and ``V``.
    kernel_YX : ndarray of shape (N, N)
        Kernel matrix between output and input data (or inducing points for Nystroem).
    kernel_Xin_X : ndarray of shape (N_eval, N)
        Kernel matrix between evaluation and input data (or inducing points for Nystroem).
    obs_train_Y : ndarray of shape (N, *)
        Observables on training output data (or inducing points for Nystroem).

    Returns
    -------
    ndarray
        Predicted observable values after ``num_steps``.
    """
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    U = fit_result["U"]
    V = fit_result["V"]
    npts = U.shape[0]
    K_dot_U = kernel_Xin_X @ U / sqrt(npts)
    V_dot_obs = V.T @ obs_train_Y / sqrt(npts)
    V_K_YX_U = np.linalg.multi_dot([V.T, kernel_YX, U]) / npts
    M = np.linalg.matrix_power(V_K_YX_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])


def pcr(
    kernel_X: ndarray,
    tikhonov_reg: float = 0.0,
    rank: int | None = None,
    svd_solver: Literal["arpack", "dense"] = "arpack",
    tol: float = 0,
    max_iter: int | None = None,
) -> FitResult:
    """Fits the Principal Components estimator.

    Args:
        kernel_X (ndarray): Kernel matrix of the input data.
        tikhonov_reg (float, optional): Tikhonov (ridge) regularization parameter.
        Defaults to 0.0.
        rank (int | None, optional): Rank of the estimator. Defaults to None.
        svd_solver (Literal[ &quot;arpack&quot;, &quot;dense&quot; ], optional):
        Solver for the generalized eigenvalue problem. Defaults to "arpack".
        tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
        max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.
    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.
    """
    npts = kernel_X.shape[0]
    add_diagonal_(kernel_X, npts * tikhonov_reg)
    if svd_solver == "arpack":
        _num_arpack_eigs = min(rank + 5, kernel_X.shape[0])
        values, vectors = eigsh(kernel_X, k=_num_arpack_eigs, maxiter=max_iter, tol=tol)
    elif svd_solver == "dense":
        values, vectors = scipy.linalg.eigh(kernel_X)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_X, -npts * tikhonov_reg)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    Q = sqrt(npts) * vectors / np.sqrt(values)
    kernel_X_eigvalsh = np.sqrt(np.abs(values)) / npts
    result: FitResult = {"U": Q, "V": Q, "svals": kernel_X_eigvalsh}
    return result


def nystroem_pcr(
    kernel_X: ndarray,
    kernel_Y: ndarray,
    kernel_Xnys: ndarray,
    kernel_Ynys: ndarray,
    tikhonov_reg: float = 0.0,
    rank: int | None = None,
    svd_solver: Literal["arpack", "dense"] = "arpack",
    tol: float = 0,
    max_iter: int | None = None,
) -> FitResult:
    """Fits the Principal Components estimator using the Nyström method
    from :footcite:t:`Meanti2023`.

    Args:
        kernel_X (ndarray): Kernel matrix of the input inducing points.
        kernel_Y (ndarray): Kernel matrix of the output inducing points.
        kernel_Xnys (ndarray): Kernel matrix between the input data and the
        input inducing points.
        kernel_Ynys (ndarray): Kernel matrix between the output data and the
        output inducing points.
        tikhonov_reg (float, optional): Tikhonov (ridge) regularization parameter.
        Defaults to 0.0.
        rank (int | None, optional): Rank of the estimator. Defaults to None.
        svd_solver (Literal[ "arpack", "dense" ], optional): Solver for the
        generalized eigenvalue problem. Defaults to "arpack".
        tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
        max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.
    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of
        training data.

        ``kernel_Y``: :math:`(N, N)`.

        ``kernel_Xnys``: :math:`(N, M)`, where :math:`M` is the number of
        Nystroem centers (inducing points).

        ``kernel_Ynys``: :math:`(N, M)`.
    """
    ncenters = kernel_X.shape[0]
    npts = kernel_Xnys.shape[0]
    eps = 1000 * np.finfo(kernel_X.dtype).eps
    reg = max(eps, tikhonov_reg)
    kernel_Xnys_sq = kernel_Xnys.T @ kernel_Xnys
    add_diagonal_(kernel_X, reg * ncenters)
    if svd_solver == "dense":
        values, vectors = scipy.linalg.eigh(
            kernel_Xnys_sq, kernel_X
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arpack":
        _oversampling = max(10, 4 * int(np.sqrt(rank)))
        _num_arpack_eigs = min(rank + _oversampling, ncenters)
        values, vectors = eigsh(
            kernel_Xnys_sq,
            M=kernel_X,
            k=_num_arpack_eigs,
            which="LM",
            maxiter=max_iter,
            tol=tol,
        )
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_X, -reg * ncenters)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]

    U = sqrt(ncenters) * vectors / np.sqrt(values)
    V = np.linalg.multi_dot([kernel_Ynys.T, kernel_Xnys, vectors])
    V = scipy.linalg.lstsq(kernel_Y, V)[0]
    V = sqrt(ncenters) * V / np.sqrt(values)

    kernel_X_eigvalsh = np.sqrt(np.abs(values)) / npts
    result: FitResult = {"U": U, "V": V, "svals": kernel_X_eigvalsh}
    return result


def rand_pcr(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int,  # Number of oversamples
    iterated_power: int,  # Number of iterations for the power method
    rng_seed: int | None = None,  # Seed for the random number generator
):
    npts = kernel_X.shape[0]
    add_diagonal_(kernel_X, npts * tikhonov_reg)
    vectors, values, _ = randomized_svd(
        kernel_X,
        rank,
        n_oversamples=n_oversamples,
        n_iter=iterated_power,
        random_state=rng_seed,
    )
    add_diagonal_(kernel_X, -npts * tikhonov_reg)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    Q = sqrt(npts) * vectors / np.sqrt(values)
    kernel_X_eigvalsh = np.sqrt(np.abs(values)) / npts
    result: FitResult = {"U": Q, "V": Q, "svals": kernel_X_eigvalsh}
    return result


def reduced_rank(
    kernel_X: ndarray,  # Kernel matrix of the input data
    kernel_Y: ndarray,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arpack", "dense"] = "arpack",
    tol: float = 0,
    max_iter: int | None = None,
) -> FitResult:
    """Fits the Reduced Rank estimator from :footcite:t:`Kostic2022`.

    Args:
        kernel_X (ndarray): Kernel matrix of the input data.
        kernel_Y (ndarray): Kernel matrix of the output data.
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter.
        rank (int): Rank of the estimator.
        svd_solver (Literal[ "arpack", "dense" ], optional): Solver for the
        generalized eigenvalue problem. Defaults to "arpack".
        tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
        max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training
        data.

        ``kernel_Y``: :math:`(N, N)`.
    """
    # Number of data points
    npts = kernel_X.shape[0]
    eps = 1000.0 * np.finfo(kernel_X.dtype).eps
    penalty = max(eps, tikhonov_reg) * npts

    A = (kernel_Y / sqrt(npts)) @ (kernel_X / sqrt(npts))
    add_diagonal_(kernel_X, penalty)
    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is
    # ill-conditioned might be slow. Prefer svd_solver == 'randomized' in
    # such a case.
    if svd_solver == "arpack":
        # Adding a small buffer to the arpack-computed eigenvalues.
        num_arpack_eigs = min(rank + 5, npts)
        values, vectors = eigs(
            A, k=num_arpack_eigs, M=kernel_X, maxiter=max_iter, tol=tol
            )
    elif svd_solver == "dense":  # 'dense'
        values, vectors = scipy.linalg.eig(
            A, kernel_X, overwrite_a=True, overwrite_b=True
            )
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    # Remove the penalty from kernel_X (inplace)
    add_diagonal_(kernel_X, -penalty)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength,
    # and warn if there are any eigenvalues that are smaller than the
    # regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        warn(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the "
            f"{len(values)} squared singular values are smaller than the "
            f"regularization strength {tikhonov_reg:.2e}. Consider reducing "
            "the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    kernel_X_vecs = np.dot(kernel_X / sqrt(npts), vectors)
    vecs_norm = np.sqrt(
        np.sum(
            kernel_X_vecs**2 + tikhonov_reg * kernel_X_vecs * vectors * sqrt(npts),
            axis=0,
        )
    )

    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    U = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]

    # Ordering the results
    V = kernel_X @ U
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def nystroem_reduced_rank(
    kernel_X: ndarray,
    kernel_Y: ndarray,
    kernel_Xnys: ndarray,
    kernel_Ynys: ndarray,
    tikhonov_reg: float,
    rank: int,
    svd_solver: Literal["arpack", "dense"] = "arpack",
    tol: float = 0,
    max_iter: int | None = None,
) -> FitResult:
    """Fits the Nyström Reduced Rank estimator from :footcite:t:`Meanti2023`.

    Args:
        kernel_X (ndarray): Kernel matrix of the input inducing points.
        kernel_Y (ndarray): Kernel matrix of the output inducing points.
        kernel_Xnys (ndarray): Kernel matrix between the input data and the
        input inducing points.
        kernel_Ynys (ndarray): Kernel matrix between the output data and the
        output inducing points.
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter.
        rank (int): Rank of the estimator.
        svd_solver (Literal[ "arpack", "dense" ], optional): Solver for the
        generalized eigenvalue problem. Defaults to "arpack".
        tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.
        max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training
        data.

        ``kernel_Y``: :math:`(N, N)`.

        ``kernel_Xnys``: :math:`(N, M)`, where :math:`M` is the number of
        Nystroem centers (inducing points).

        ``kernel_Ynys``: :math:`(N, M)`.
    """
    num_points = kernel_Xnys.shape[0]
    num_centers = kernel_X.shape[0]

    eps = 1000 * np.finfo(kernel_X.dtype).eps * num_centers
    reg = max(eps, tikhonov_reg)

    # LHS of the generalized eigenvalue problem
    sqrt_Mn = sqrt(num_centers * num_points)
    kernel_YX_nys = (kernel_Ynys.T / sqrt_Mn) @ (kernel_Xnys / sqrt_Mn)

    _tmp_YX = scipy.linalg.lstsq(kernel_Y * (num_centers**-1), kernel_YX_nys)[0]
    kernel_XYX = kernel_YX_nys.T @ _tmp_YX

    # RHS of the generalized eigenvalue problem
    kernel_Xnys_sq = (kernel_Xnys.T / sqrt_Mn) @ (kernel_Xnys / sqrt_Mn)
    + reg * kernel_X * (num_centers**-1)

    add_diagonal_(kernel_Xnys_sq, eps)
    A = scipy.linalg.lstsq(kernel_Xnys_sq, kernel_XYX)[0]
    if svd_solver == "dense":
        values, vectors = scipy.linalg.eigh(
            kernel_XYX, kernel_Xnys_sq
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arpack":
        _oversampling = max(10, 4 * int(np.sqrt(rank)))
        _num_arpack_eigs = min(rank + _oversampling, num_centers)
        values, vectors = eigs(
            kernel_XYX,
            k=_num_arpack_eigs,
            M=kernel_Xnys_sq,
            maxiter=max_iter,
            tol=tol
            )
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_Xnys_sq, -eps)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn
    # if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        warn(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the "
            f"{len(values)} squared singular values are smaller than the "
            f"regularization strength {tikhonov_reg:.2e}. Consider reducing the "
            "regularization strength to avoid overfitting."
        )
    # Eigenvector normalization
    vecs_norm = np.sqrt(np.abs(np.sum(vectors.conj() * (kernel_XYX @ vectors), axis=0)))
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
    U = A @ vectors
    V = _tmp_YX @ vectors
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def rand_reduced_rank(
    kernel_X: ndarray,
    kernel_Y: ndarray,
    tikhonov_reg: float,
    rank: int,
    n_oversamples: int = 5,
    optimal_sketching: bool = False,
    iterated_power: Literal["auto"] | int = "auto",
    rng_seed: int | None = None,
    precomputed_cholesky=None,
) -> FitResult:
    """Fits the Randomized Reduced Rank Estimator from :footcite:t:`Turri2023`.

    Args:
        kernel_X (ndarray): Kernel matrix of the input data
        kernel_Y (ndarray): Kernel matrix of the output data
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter
        rank (int): Rank of the estimator
        n_oversamples (int, optional): Number of Oversamples. Defaults to 5.
        optimal_sketching (bool, optional): Whether to use optimal sketching
        (slower but more accurate) or not. Defaults to False.
        iterated_power (int, optional): Number of iterations of the power method.
        Defaults to 1.
        rng_seed (int | None, optional): Random Number Generators seed. Defaults
        to None.
        precomputed_cholesky (optional): Precomputed Cholesky decomposition.
        Should be the output of cho_factor evaluated on the regularized kernel
        matrix. Defaults to None.

    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.

        ``kernel_Y``: :math:`(N, N)`.
    """
    rng = np.random.default_rng(rng_seed)
    npts = kernel_X.shape[0]

    penalty = npts * tikhonov_reg
    add_diagonal_(kernel_X, penalty)
    if precomputed_cholesky is None:
        cholesky_decomposition = scipy.linalg.cho_factor(kernel_X)
    else:
        cholesky_decomposition = precomputed_cholesky
    add_diagonal_(kernel_X, -penalty)

    sketch_dimension = rank + n_oversamples

    if optimal_sketching:
        cov = kernel_Y / npts
        sketch = rng.multivariate_normal(
            np.zeros(npts, dtype=kernel_Y.dtype), cov, size=sketch_dimension
        ).T
    else:
        sketch = rng.standard_normal(size=(npts, sketch_dimension))

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        sketch = (kernel_Y / npts) @ (
            sketch - penalty * scipy.linalg.cho_solve(cholesky_decomposition, sketch)
        )
        sketch, _ = scipy.linalg.qr(sketch, mode="economic")  # QR re-orthogonalization

    kernel_X_sketch = scipy.linalg.cho_solve(cholesky_decomposition, sketch)
    _M = sketch - penalty * kernel_X_sketch

    F_0 = sketch.T @ sketch - penalty * (sketch.T @ kernel_X_sketch)  # Symmetric
    F_0 = 0.5 * (F_0 + F_0.T)
    F_1 = _M.T @ ((kernel_Y @ _M) / npts)

    values, vectors = scipy.linalg.eig(scipy.linalg.lstsq(F_0, F_1)[0])
    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]

    # Remove elements in the kernel of F_0
    relative_norm_sq = np.abs(
        np.sum(vectors.conj() * (F_0 @ vectors), axis=0)
        / np.linalg.norm(vectors, axis=0) ** 2
    )
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(relative_norm_sq, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs]

    vecs_norms = (np.sum(vectors.conj() * (F_0 @ vectors), axis=0).real) ** 0.5
    vectors = vectors / vecs_norms

    U = sqrt(npts) * kernel_X_sketch @ vectors
    V = sqrt(npts) * _M @ vectors
    svals = np.sqrt(values)
    result: FitResult = {"U": U, "V": V, "svals": svals}
    return result
