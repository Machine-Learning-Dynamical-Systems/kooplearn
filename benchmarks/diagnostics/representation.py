# 0091/benchmarks/diagnostics/representation.py
# ===========================================================================
# Representation-sensitive spectral diagnostics, including the implementation
# of Kostic et al. 2023 metrics for analysing kernel spectrums.
# This file includes:
# - metric distortion;
# - spectral bias (including the truncation functions for each estimator);
# - spectral gap (as defined in Kostic 2023 Appendix C);
# - SSL criteria (spectral contrastive loss and VAMP-2,
#   both are per-data-draw scalars).
# ===========================================================================

import numpy as np

from kooplearn._linalg import eigh_rank_reveal, spd_neg_pow, weighted_norm


# ==========================
# --- metric distortion ---
# ==========================
def metric_distortion(psi, C):
    r"""Empirical metric distortion :math:`\widehat\eta_i = \|\widehat\psi_i\|_{\mathcal H} /
    \sqrt{\langle \widehat C \widehat\psi_i, \widehat\psi_i\rangle}`.

    Parameters
    ----------
    psi : ndarray, shape (n,) or (n, k)
        Eigenfunction(s) evaluated at the *training* points. If 2D, each
        column is treated as a separate eigenfunction (see `weighted_norm`).
    C : ndarray, shape (n, n)
        Empirical (kernel-based) covariance, i.e. ``model.kernel_X / n_samples``.
    """
    psi = np.asarray(psi)
    n = C.shape[0]

    # ||psi||_H via the reproducing property: needs the *inverse* Gram, since
    # C = K_X / n is the Gram-based covariance, not the RKHS metric itself.
    C_inv = spd_neg_pow(C * n, exponent=-1.0)  # i.e. K_X^{-1}
    rkhs_norm = weighted_norm(psi, M=C_inv)

    # <C psi, psi> = (1/n)||psi(X)||_2^2, i.e. weighted_norm with M=None, squared, over n
    empirical_norm = weighted_norm(psi, M=None) / np.sqrt(n)

    with np.errstate(divide="ignore", invalid="ignore"):
        eta = rkhs_norm / empirical_norm
    eta = np.where(empirical_norm > 0, eta, np.nan)
    return eta if psi.ndim == 2 else float(eta)


# =====================
# --- spectral bias ---
# =====================


# --- truncation helpers ---


def _top_sv(C, r):
    """(r+1)-st eigenvalue of a symmetric PSD matrix, via eigh_rank_reveal."""
    raw_vals, raw_vecs = np.linalg.eigh(np.asarray(C))
    _, top_vals, _ = eigh_rank_reveal(raw_vals, raw_vecs, rank=r + 1)
    if len(top_vals) <= r:
        return 0.0
    return float(top_vals[-1])


def pcr_truncation(C, r):
    r""":math:`\rho_{r+1}(\widehat G^{PCR}) = \sigma_{r+1}(\widehat C)`."""
    return _top_sv(C, r)


# kDMD uses the same (r+1)-st eigenvalue of the empirical covariance as PCR
kdmd_truncation = pcr_truncation


def rrr_truncation(C, T, r, cutoff=None):
    r""":math:`\rho_{r+1}(\widehat G^{RRR}) = \sigma_{r+1}(\widehat C^{-1/2}\widehat T)`."""
    C_inv_sqrt = spd_neg_pow(np.asarray(C), exponent=-0.5, cutoff=cutoff)
    A = C_inv_sqrt @ np.asarray(T)
    svals = np.linalg.svd(A, compute_uv=False)
    if r >= len(svals):
        return 0.0
    return float(svals[r])


# --- bias function ---


def spectral_bias(eigenfunction, C, rho):
    r"""Empirical spectral bias :math:`\hat s_i = \widehat\eta_i \, \rho_{r+1}`."""
    eta = metric_distortion(eigenfunction, C)
    s_hat = eta * rho
    return float(s_hat), eta


# ====================================================
# --- spectral gap (|mu_l - mu_j|) ---
# ====================================================
def koopman_gap(eigs):
    """gap_j = min_{l != j} |mu_l - mu_j| for each eigenvalue in `eigs`.

    Call on the full vals_hat inside the experiment loops
    (all n_components), record per mode alongside spectral_bias.
    """
    eigs = np.atleast_1d(np.asarray(eigs, complex))
    n = len(eigs)
    if n < 2:
        return np.full(n, np.nan)
    D = np.abs(eigs[:, None] - eigs[None, :])
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


#   * horizon terms need the trial *dispersion* of each eigenvalue, so they are
#     candidate-level by nature (one value per kernel x kind x method, not per
#     trial).  Compute them once per run from the same per-mode records that
#     feed analyse_spectrum, then hand the small table to kernel_spectral_score
#     via `extra_metrics=` (still one call, still within the experiment).


# ===========================================================================
# --- SSL selection criteria: spectral-contrastive loss & VAMP-2 ------------
# ===========================================================================
R_VAMP = 3  # VAMP-2 rank (modes 1..3)
N_VAMP = 400  # subsample for the whitened SVD (n^3 cost)
EPS_WHITEN = 1e-6  # ridge on the whitening eigenvalues


# This was the implementation following HaoChen et al. 2021;
# this version takes P=I, and is not theoretically accurate.
# Included here as a log of the progress made only.
def sc_criterion(K_xy_val):
    r"""Spectral-contrastive loss from the cross-kernel matrix on lagged
    validation pairs (lower = better).

    .. math::

        \mathrm{SC} = \frac{1}{N(N-1)}\sum_{i\neq j} K_{ij}^2
                      - \frac{2}{N}\sum_i K_{ii},

    with :math:`K = K(X_\text{val}, Y_\text{val})` the cross-kernel Gram matrix.
    """
    K = np.asarray(K_xy_val, float)
    N = K.shape[0]
    diag = np.diag(K)
    off_sq = (K**2).sum() - (diag**2).sum()
    return float(off_sq / (N * (N - 1)) - 2.0 * diag.mean())


def vamp2_r(K_X, K_Y, K_XY, r=R_VAMP, eps=EPS_WHITEN):
    r"""VAMP-2(r): sum of the top-``r`` squared singular values of the whitened
    cross-operator, estimated from Gram matrices (higher = better).

    .. math::

        \mathrm{VAMP2}_r = \sum_{k=1}^{r} \sigma_k^2\!\big(
            K_X^{-1/2}\, K_{XY}\, K_Y^{-1/2}\big),

    the singular values being the estimated canonical correlations
    (clipped to :math:`[0, 1]`).  ``eps`` ridges the whitening eigenvalues.
    """
    n = K_X.shape[0]

    def inv_sqrt(K):
        w, V = np.linalg.eigh(K / n)
        w = np.clip(w, eps, None)
        return (V / np.sqrt(w)) @ V.T / np.sqrt(n)

    A = inv_sqrt(K_X) @ K_XY @ inv_sqrt(K_Y)
    s = np.linalg.svd(A, compute_uv=False)
    s = np.clip(s, 0, 1.0 + 1e-6)  # canonical correlations
    return float(np.sum(s[:r] ** 2))


# This is the correct optimal predictor formulation (P*-VAMP)
def vamp2_ridge(K_X, K_Y, lam=None):
    r"""Ridge/Tikhonov VAMP-2 (full-rank trace form) from the two AUTO-Gram
    matrices alone -- no cross-kernel matrix needed (higher = better).

    .. math::

        \mathrm{VAMP2}_\lambda = \operatorname{tr}\big[(K_X+\lambda I)^{-1}K_X\,
                                   (K_Y+\lambda I)^{-1}K_Y\big]

    By the push-through identity :math:`(X^\top X+\lambda I)^{-1}X^\top
    = X^\top(XX^\top+\lambda I)^{-1}` this equals the primal
    :math:`\operatorname{tr}[(C_X+\lambda I)^{-1}C_{XY}(C_Y+\lambda I)^{-1}C_{YX}]`
    exactly (no approximation; verified to machine precision in the tests).
    The cross-domain information lives entirely in the shared row pairing of
    :math:`K_X` and :math:`K_Y`, so no explicit :math:`K_{XY}` is required.
    Loss convention (Lemma 3): ``loss = -vamp2_ridge``.

    Differences from :func:`vamp2_r` (the truncated form):

    * **full-rank** -- sums ridge-shrunk squared canonical correlations over
      *all* components, not the top ``r``; disagreement between the two
      quantifies how much predictable signal lives outside rank ``r``;
    * **no clipping** -- regularisation is by the ridge :math:`\lambda`
      (shrinkage :math:`\sigma_i^2 \to` weighted by :math:`k/(k+\lambda)`
      factors) instead of clipping canonical correlations to ``[0, 1]``;
    * needs no SVD and no ``K_XY`` -- one solve per Gram matrix.

    ``lam=None`` defaults to ``EPS_WHITEN * n``, the same absolute ridge scale
    that ``eps`` applies to the spectrum of ``K/n`` inside :func:`vamp2_r`.
    """
    K_X = np.asarray(K_X, float)
    K_Y = np.asarray(K_Y, float)
    n = K_X.shape[0]
    if lam is None:
        lam = EPS_WHITEN * n
    I = np.eye(n)
    return float(
        np.trace(
            np.linalg.solve(K_X + lam * I, K_X) @ np.linalg.solve(K_Y + lam * I, K_Y)
        )
    )


def vamp2_ridge_features(phi_X, phi_Y, lam=None):
    r"""Primal (feature-space) twin of :func:`vamp2_ridge` for explicit
    encoders: :math:`\operatorname{tr}[(C_X+\lambda I)^{-1}C_{XY}
    (C_Y+\lambda I)^{-1}C_{YX}]` on the ``D x D`` covariances -- identical
    value to the Gram form by push-through, at :math:`O(N D^2)` cost."""
    X = np.asarray(phi_X, float)
    Y = np.asarray(phi_Y, float)
    n, d = X.shape
    if lam is None:
        lam = EPS_WHITEN * n
    I = np.eye(d)
    Cx, Cy, Cxy = X.T @ X, Y.T @ Y, X.T @ Y
    return float(
        np.trace(
            np.linalg.solve(Cx + lam * I, Cxy) @ np.linalg.solve(Cy + lam * I, Cxy.T)
        )
    )


def vamp2_ridge_sweep(K_X, K_Y, lams):
    r"""VAMP-2 along the ridge-predictor axis: ``[vamp2_ridge(K_X, K_Y, lam)]``
    for each ``lam`` in ``lams``.

    Traces the family from the (near-)exact optimal predictor :math:`P_*`
    (``lam -> 0``, full unregularised VAMP-2 = ``-sc_at_optimal_P``) to a heavily
    shrunk predictor (``lam`` large).  Compute this per candidate inside the
    experiment loop, then plot ``Spearman(vamp2_ridge(lam), true_eig_err)`` vs
    ``lam`` to show WHERE on the P_*<->regularised axis the selection signal lives
    (does it degrade as ``lam -> 0`` toward the saturating P_*?).  Needs the Gram
    matrices, so run it in the experiment (they are not stored in the score CSVs).
    Feature-space encoders: use :func:`vamp2_ridge_features` in the same loop.
    """
    return np.array([vamp2_ridge(K_X, K_Y, lam) for lam in np.atleast_1d(lams)])


# Turri et al., 2026 formulation of Spectral Contrastive at P*
def sc_at_optimal_P(K_X, K_Y, lam=None):
    r"""Spectral-contrastive loss evaluated at the OPTIMAL predictor
    :math:`P_*` (Lemma 3, ICLR-2026 SSL paper): exactly

    .. math::

        \varepsilon(\varphi, P_*) = -\big\|C_X^{-1/2} C_{XY} C_Y^{-1/2}
        \big\|_{\mathrm{HS}}^2 = -\mathrm{VAMP}_2(\varphi),

    i.e. the negative FULL-RANK, unclipped VAMP-2, computed here in the
    Gram/dual form ``-vamp2_ridge(K_X, K_Y, lam)`` (ridge :math:`\lambda`
    for numerical stability; converges to the exact identity as
    :math:`\lambda \to 0`; verified to machine precision in the tests).
    """
    return -vamp2_ridge(K_X, K_Y, lam)


def sc_at_optimal_P_features(phi_X, phi_Y, lam=None):
    r"""Feature-regime twin of :func:`sc_at_optimal_P`:
    :math:`-`:func:`vamp2_ridge_features` (identical value by push-through)."""
    return -vamp2_ridge_features(phi_X, phi_Y, lam)


def hs_gap(K_X, K_Y, lam=None):
    r"""Deflated HS-distance criterion (Oberwolfach slides / Kostic-Lounici
    NeurIPS 2024; Turri et al. ICLR 2026), Gram/dual regime (lower = better).

    The slides' object is :math:`\|\mathsf{E}_{X|Y}-\mathsf{E}_\theta\|^2_{HS}
    = \mathcal{L}(\theta) + c` with :math:`c = \|\mathsf{E}_{X|Y}\|^2_{HS}`
    candidate-independent, and the model DEFLATED by the constant mode
    (:math:`\mathsf{E} = 1\otimes 1 + \sum_i \sigma_i u_i \otimes v_i`).
    At the optimal model the loss attains :math:`-\sum_i \sigma_i^2`, i.e.
    minus the **centered** (mean-deflated) full-rank VAMP-2.  Hence, for
    selection purposes (ranking within a system), the HS distance is
    equivalent to this function:

    .. math::

        \mathrm{hs\_gap} = -\operatorname{tr}\big[(\tilde K_X+\lambda I)^{-1}
        \tilde K_X\,(\tilde K_Y+\lambda I)^{-1}\tilde K_Y\big],
        \qquad \tilde K = H K H,\; H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top .

    Relation to :func:`sc_at_optimal_P`: identical except for the deflation --
    ``sc_star`` retains the trivial constant mode (every reasonable kernel
    captures it, inflating all scores by :math:`\approx 1` and masking
    differences); ``hs_gap`` removes it, which is the paper's object.
    """
    K_X = np.asarray(K_X, float)
    K_Y = np.asarray(K_Y, float)
    n = K_X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return -vamp2_ridge(H @ K_X @ H, H @ K_Y @ H, lam)


def hs_gap_features(phi_X, phi_Y, lam=None):
    r"""Feature-regime twin of :func:`hs_gap`: mean-center the features, then
    :math:`-`:func:`vamp2_ridge_features` (identical value by push-through)."""
    X = np.asarray(phi_X, float)
    Y = np.asarray(phi_Y, float)
    return -vamp2_ridge_features(X - X.mean(0), Y - Y.mean(0), lam)


# Compute both SCL and VAMP in one go
def kernel_ssl_criteria(
    K_fn, X_tr, Y_tr, X_val, Y_val, *, r=R_VAMP, n_vamp=N_VAMP, eps=EPS_WHITEN
):
    r"""Both SSL criteria for ONE kernel on ONE data draw.

    ``K_fn(A, B)`` must return the Gram matrix between row-sets ``A`` and ``B``.
    VAMP-2 uses a size-``n_vamp`` subsample of the training pairs (the whitening
    SVD is :math:`O(n^3)`); SC uses the full validation pairs.

    Returns ``{"sc": ..., "vamp2": ...}`` -- merge into the trial record so the
    grand scorer picks them up as per-trial ``sc`` / ``vamp2`` columns.
    """
    sub = slice(0, n_vamp)
    K_x = K_fn(X_tr[sub], X_tr[sub])
    K_y = K_fn(Y_tr[sub], Y_tr[sub])
    K_xy = K_fn(X_tr[sub], Y_tr[sub])
    return {
        "sc": sc_criterion(K_fn(X_val, Y_val)),
        "vamp2": vamp2_r(K_x, K_y, K_xy, r=r, eps=eps),
        "vamp2_ridge": vamp2_ridge(K_x, K_y),
        "sc_star": sc_at_optimal_P(K_x, K_y),
        "hs_gap": hs_gap(K_x, K_y),
    }
