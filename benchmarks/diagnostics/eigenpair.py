# 0091/benchmarks/diagnostics/candidate.py
# ===========================================================================
# Mode-level diagnostics: spectral verification and stability *per-mode*
# This file includes: ResDMD residual (including Galerkin matrices helper)
# ===========================================================================

import numpy as np
import pandas as pd
from kooplearn._linalg import weighted_norm


# ===========================================================================
# --- ResDMD residual & pseudospectra (Colbrook & Townsend 2024) ------------
# ===========================================================================
#
# THE residual used throughout the pipeline is the ResDMD residual.  For a
# candidate eigenpair (lambda, g), with Galerkin matrices G = Psi_X^* W Psi_X,
# A = Psi_X^* W Psi_Y and the third matrix L = Psi_Y^* W Psi_Y,
#
#   res(lambda, g)^2 = g^* [ L - lambda A^* - conj(lambda) A + |lambda|^2 G ] g
#                      / ( g^* G g )                                (Colbrook 3.2)
#
# converges to the true operator residual ||K g - lambda g|| / ||g|| and, by
# Thm 3.1, upper-bounds the distance of lambda to the true spectrum -- a *verified*
# pseudospectral certificate.  Equivalently, in function-value form,
#   res = ||psi(Y) - lambda psi(X)|| / ||psi(X)||,
# which is IDENTICAL for eigenpairs (verified to ~2e-15; the third matrix enters
# as ||psi(Y)||^2 = g^* L g).  There is ONE residual function, `resdmd_residual`,
# computing it either way.  Algorithm 1 accepts eigenpairs with res <= eps (a
# pseudospectral tolerance); `resdmd_pseudospectrum` (Algorithm 2) extends the
# same residual to arbitrary z to trace the pseudospectrum (non-normal systems).


def galerkin_matrices(Phi_X, Phi_Y, weights=None):
    r"""The three ResDMD Galerkin matrices from dictionary evaluations.

    Returns ``(G, A, L)`` with ``G = Psi_X^* W Psi_X``, ``A = Psi_X^* W Psi_Y``,
    ``L = Psi_Y^* W Psi_Y``.  ``Phi_X`` / ``Phi_Y`` are the dictionary/features on
    inputs and one-step outputs, shape ``(n, d)``.  ``L`` is the one extra matrix
    ResDMD needs beyond EDMD (costs nothing -- ``Phi_Y`` is already in hand); it is
    used implicitly by the function-value residual as ``||psi(Y)||^2 = g^* L g``.
    """
    Phi_X = np.asarray(Phi_X)
    Phi_Y = np.asarray(Phi_Y)
    n = Phi_X.shape[0]
    w = (
        np.full(n, 1.0 / n)
        if weights is None
        else np.asarray(weights, float) / np.sum(weights)
    )
    WXt = Phi_X.conj().T * w  # (d, n): columns scaled by the quadrature weights
    G = WXt @ Phi_X
    A = WXt @ Phi_Y
    L = (Phi_Y.conj().T * w) @ Phi_Y
    return G, A, L


def resdmd_residual(
    eigenvalues,
    psi_X=None,
    psi_Y=None,
    eps=None,
    relative=True,
    *,
    eigvecs=None,
    G=None,
    A=None,
    L=None,
):
    r"""THE ResDMD residual per eigenpair (Colbrook eq. 3.2) -- the single residual
    used everywhere in the pipeline.  Computed in either of two mathematically
    identical ways (verified to ~2e-15):

    * **function-value form (default)** -- pass ``psi_X`` / ``psi_Y`` = the
      eigenfunctions evaluated at inputs / one-step outputs, shape ``(n, r)``.
      ``res_i = ||psi_i(Y) - lambda_i psi_i(X)|| / ||psi_i(X)||``.  This is what the
      experiment loops have to hand, so it is the efficient default.
    * **Galerkin form** -- pass ``eigvecs`` ``(d, r)`` and ``G, A, L`` from
      :func:`galerkin_matrices` to evaluate the explicit three-matrix quadratic
      form 3.2 (needed when the features, not the eigenfunction values, are held).

    Returns
    -------
    res : ndarray (r,)                       if ``eps`` is None.
    (n_spurious, res) : (int, ndarray)       if ``eps`` is given -- the ResDMD
        Algorithm-1 result, ``n_spurious = #{res > eps}`` (``eps`` = pseudospectral
        tolerance; lower res = better verified).
    """
    vals = np.atleast_1d(np.asarray(eigenvalues))
    if G is not None:  # explicit Galerkin form
        V = np.asarray(eigvecs)
        res = np.empty(V.shape[1], dtype=float)
        for i in range(V.shape[1]):
            lam = vals[i]
            g = V[:, i]
            R = L - lam * A.conj().T - np.conj(lam) * A + np.abs(lam) ** 2 * G
            num = (g.conj() @ R @ g).real
            den = (g.conj() @ G @ g).real
            res[i] = np.sqrt(max(num, 0.0) / den) if den > 0 else np.nan
    else:  # function-value form (identical for eigenpairs)
        n = psi_X.shape[0]
        resid = np.asarray(psi_Y) - np.asarray(psi_X) * vals[None, :]
        resid_norm = weighted_norm(resid) / np.sqrt(n)
        if relative:
            base = weighted_norm(psi_X) / np.sqrt(n)
            res = np.full_like(resid_norm, np.nan, dtype=float)
            ok = np.isfinite(base) & (base > 0)
            res[ok] = resid_norm[ok] / base[ok]
        else:
            res = resid_norm
    if eps is None:
        return res
    with np.errstate(invalid="ignore"):
        n_spurious = int(np.sum(res > eps))  # NaN > eps -> False, so NaNs are ignored
    return n_spurious, res


def resdmd_verified_mask(residual, eps):
    """Colbrook Algorithm-1 acceptance mask: ``True`` where ``residual <= eps``."""
    return np.asarray(residual) <= eps


# Backward-compatible alias -- existing experiment code calls
# `spurious_residual(eigs, psi_X, psi_Y, delta)`.  It IS `resdmd_residual`
# (function-value form); `delta` is the pseudospectral tolerance `eps`.  Prefer
# `resdmd_residual` in new code.
spurious_residual = resdmd_residual


def resdmd_pseudospectrum(z_grid, G, A, L):
    r"""ResDMD pseudospectrum (Colbrook Algorithm 2): ``tau(z) = min_g res(z, g)``.

    For each complex ``z``, ``tau(z)`` is the square root of the smallest
    generalised eigenvalue of the Hermitian pencil ``(D(z), G)`` with
    ``D(z) = L - z A^* - conj(z) A + |z|^2 G``.  ``z`` lies in the
    ``epsilon``-pseudospectrum iff ``tau(z) <= epsilon`` -- the verified region
    within which a true eigenvalue must lie.  Accepts a scalar or array of ``z``.
    """
    from scipy.linalg import eigh

    z_grid = np.atleast_1d(z_grid)
    G_h = (G + G.conj().T) / 2
    tau = np.empty(z_grid.shape, dtype=float)
    for idx, z in np.ndenumerate(z_grid):
        D = L - z * A.conj().T - np.conj(z) * A + np.abs(z) ** 2 * G
        D = (D + D.conj().T) / 2
        ev = eigh(D, G_h, eigvals_only=True)
        tau[idx] = np.sqrt(max(float(ev.min()), 0.0))
    return tau if tau.shape != (1,) else float(tau[0])


# ===========================
# --- operator norm error ---
# ===========================
def operator_norm_error(true_operator: np.ndarray, estimated_operator: np.ndarray):
    r"""Operator norm error proxy for a Koopman estimator.

    Computes the operator norm discrepancy between the true action
    :math:`A_\pi S` and the estimated action :math:`S \widehat{G}`:

    .. math::

        \mathcal{E}(\widehat{G}) := \|A_\pi S - S \widehat{G}\|.

    Since "kooplearn" does not currently expose :math:`A_\pi` or the embedding
    operator :math:`S` explicitly, this function works with their actions on a
    common finite-dimensional representation. In practice, the caller should pass
    matrices or vectors representing the two quantities to be compared.
    """
    true_operator = np.asanyarray(true_operator)
    estimated_operator = np.asanyarray(estimated_operator)

    if true_operator.shape != estimated_operator.shape:
        raise ValueError(
            "true_operator and estimated_operator must have the same "
            f"shape, got {true_operator.shape} and "
            f"{estimated_operator.shape}."
        )

    diff = true_operator - estimated_operator
    if diff.ndim == 1:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff, ord=2))


# =========================================
# --- spurious eigenvalues vs reference ---
# =========================================


def spurious_ref(est, ref, delta):
    dist = np.abs(est[:, None] - ref[None, :])
    return int(np.sum(dist.min(axis=1) > delta))


# ---- defaults (match the notebooks) ---------------------------------------
MODE_W = {1: 0.5, 2: 0.3, 3: 0.2}  # eigenfunction weights, horizon notebook
H = 50  # forecast horizon in lag steps
R_VAMP = 3  # VAMP-2 rank (modes 1..3)
N_VAMP = 400  # subsample for the whitened SVD (n^3 cost)
EPS_WHITEN = 1e-6  # ridge on the whitening eigenvalues


# ===========================================================================
# --- long-horizon stability / sensitivity ----------------------------------
# ===========================================================================
def horizon_mode_terms(lam, sigma, H=H):
    r"""Reference-free long-horizon terms for ONE eigenmode.

    Given the trial-mean estimated eigenvalue :math:`\hat\lambda` and its
    trial dispersion :math:`\sigma`, over a forecast horizon of ``H`` lag steps:

    .. math::

        \text{instab} = \max(0,\ |\hat\lambda| - 1), \qquad
        \text{sens}   = \frac{1}{H}\sum_{t=1}^{H} t\,|\hat\lambda|^{\,t-1}\,\sigma .

    ``instab`` penalises spectra that blow up (magnitude > 1); ``sens`` is the
    mean derivative of the ``t``-step propagator w.r.t. eigenvalue error, i.e.
    how fast trial noise is amplified along the horizon.
    """
    t = np.arange(1, H + 1)
    instab = max(0.0, abs(lam) - 1.0)
    sens = float(np.mean(t * np.abs(lam) ** (t - 1) * sigma))
    return instab, sens


def horizon_terms(
    modes_df, H=H, mode_weights=MODE_W, group_cols=("kernel", "kind", "method")
):
    r"""Aggregate long-horizon terms per candidate from per-mode trial records.

    ``modes_df`` is the per-mode metrics table (the same records handed to
    ``analyse_spectrum``); it must contain ``eigenfunction_id``,
    ``est_eig_real``, ``est_eig_imag`` and the ``group_cols``.  For each mode
    the trial-mean eigenvalue and trial dispersion are formed, converted to
    per-mode terms via :func:`horizon_mode_terms`, then combined as a
    weighted average over modes.

    Returns a DataFrame with ``group_cols`` + ``agg_horizon_instab`` +
    ``agg_horizon_sens`` (one row per candidate) -- pass straight to
    ``kernel_spectral_score(..., extra_metrics=...)``.
    """
    group_cols = list(group_cols)
    df = modes_df[modes_df["eigenfunction_id"].isin(mode_weights)].copy()
    df["est"] = df["est_eig_real"] + 1j * df["est_eig_imag"]

    rows = []
    for keys, g in df.groupby(group_cols):
        instab = sens = wsum = 0.0
        for mode, gm in g.groupby("eigenfunction_id"):
            w = mode_weights[mode]
            lam = gm["est"].mean()  # trial-mean eigenvalue
            sigma = (
                np.sqrt(
                    gm["est_eig_real"].std() ** 2  # trial dispersion
                    + gm["est_eig_imag"].std() ** 2
                )
                if len(gm) > 1
                else 0.0
            )
            di, ds = horizon_mode_terms(lam, sigma, H=H)
            instab += w * di
            sens += w * ds
            wsum += w
        if wsum > 0:
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                dict(
                    zip(group_cols, keys),
                    agg_horizon_instab=instab / wsum,
                    agg_horizon_sens=sens / wsum,
                )
            )
    return pd.DataFrame(rows)
