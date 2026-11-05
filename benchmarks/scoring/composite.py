# 0091/benchmarks/scoring/composite.py
# ===========================================================================
# ------------------------ THE scoring configuration ------------------------
# ===========================================================================
# Please see Chapter 3.1 + Appendix A of the thesis for full details
# on the computation. This function is organised to first aggregate
# ResDMD and the Kostic et al. 2023 geometric + stability metrics,
# before adding on the long-horizon and SSL axes. This is reflective
# only of the research + experiment process of the project, not of the
# conceptual organisation of these diagnostics.
# ===========================================================================


import numpy as np
import pandas as pd
from collections.abc import Mapping


RESID_EPS = 0.1  # pseudospectral tolerance epsilon for the ResDMD residual certificate

GRAND_WEIGHTS = {
    # Six geometric-fidelity axes.  The graded residual QUALITY is the residual
    # magnitude (agg_spurious_mean/std, weighted); residual/reference COUNTS are
    # the ResDMD Algorithm-1 accept/reject FILTER (gates below).
    "agg_bias_mean": 1.0,
    "agg_dist_mean": 1.0,
    "agg_spurious_mean": 1.0,  # mean ResDMD residual magnitude (graded quality)
    "agg_spurious_std": 0.25,
    "agg_koopman_gap": 0.5,  # spectral gap
    "agg_horizon_sens": 1.0,  # long-horizon sensitivity
    "sc_star": 1.0,  # sc_star = eps(phi, P*) = -VAMP-2
}

LARGER_IS_BETTER = set()

# Metrics that are log-normalised, see below
LOG_METRICS = {
    "agg_bias_mean",
    "agg_dist_mean",
    "agg_spurious_std",
    "agg_horizon_sens",
}

# Admissibility gates (3): ResDMD residual + reference counts (spectral
# pollution) and horizon-instability blow-up flag.
HARD_CONSTRAINTS = {
    "max_spurious_ref_count": 4,
    "max_spurious_residual_count": 5,  # res > RESID_EPS
    "max_horizon_instab": 0.01,  # flag |lambda| > 1 blow-up
}

# Sentinel to distinguish
# "caller did not pass hard_constraints" --> use defaults; vs
# "caller explicitly passed None / {}" --> disable.
_DEFAULT = object()


# ===========================================================================
# --- normalisation (log -> grouped z-score) -----------
# ===========================================================================
def _normalise_series(s, method="zscore", larger_is_better=False, log=False):
    """Normalise ONE series (a metric column, or a single pool's slice of it).

    Pipeline mirrors grand_kernel_score._norm:
        to_numeric -> drop +/-inf -> optional log10(clip 1e-16)
                   -> optional sign flip (larger-is-better) -> z/minmax/rank.

    Degenerate groups (all-NaN, or zero / non-finite spread) collapse to 0.0,
    i.e. a *neutral* contribution.  All three methods are oriented so that a
    larger normalised value means "worse" (for lower-score-better).
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)

    if log:
        x = np.log10(x.clip(lower=1e-16))
    if larger_is_better:
        x = -x

    m = x.notna()
    out = pd.Series(0.0, index=x.index, dtype=float)
    if m.sum() == 0:
        return out

    vals = x[m].to_numpy(dtype=float)

    if method == "zscore":
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)  # population std (ddof=0), as in the notebook
        if np.isfinite(sd) and sd > 0:
            out.loc[m] = (vals - mu) / sd
        # else: leave neutral zeros

    elif method == "minmax":
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if np.isfinite(hi - lo) and hi > lo:
            out.loc[m] = (vals - lo) / (hi - lo)

    elif method == "rank":
        r = pd.Series(vals).rank(method="average").to_numpy(dtype=float)
        denom = r.max() - 1.0
        if denom > 0:
            out.loc[m] = (r - 1.0) / denom  # ascending: larger value -> worse

    else:
        raise ValueError(f"Unknown normalise method: {method}")

    return out


# Pooled
def _normalise_grouped(df, metric, pool_cols, *, method, larger_is_better, log):
    """Apply `_normalise_series` within each pool (or globally if no pool)."""
    kwargs = dict(method=method, larger_is_better=larger_is_better, log=log)
    if pool_cols:
        return df.groupby(list(pool_cols), dropna=False)[metric].transform(
            lambda col: _normalise_series(col, **kwargs)
        )
    return _normalise_series(df[metric], **kwargs)


# ===========================================================================
# ----------------------- guarded composite ----------------------------------
# ===========================================================================
def composite_score(
    summary,
    trials_df=None,
    *,
    group_cols=("kernel", "kind", "method"),
    candidate_col="kernel",
    pool_cols=None,
    selected_modes=None,
    mode_weights=None,
    normalise="zscore",
    metric_weights=None,
    log_metrics=LOG_METRICS,
    larger_is_better=None,
    hard_constraints=_DEFAULT,
    extra_metrics=None,
    use_trial_metrics=True,
    score_name="grand_score",
):
    """Score kernels along the selection hierarchy: mode-level and representation-level".

    Parameters
    ----------
    summary : DataFrame
        Per-mode table from `analyse_spectrum` (one row per
        kernel x kind x method x eigenfunction_id) with columns
        bias_mean/dist_mean/spurious_mean/spurious_std.
    trials_df : DataFrame, optional
        Per-trial table.  Any of these columns, if present, are aggregated to
        per-candidate axes: spurious_ref_count, spurious_residual_count,
        spectral_gap, rank, horizon_instab, horizon_sens, sc, vamp2.
    group_cols : tuple
        Candidate granularity (one scored row per unique combination).
    candidate_col : str
        The axis that *varies within a pool* (the thing being selected between).
    pool_cols : tuple, optional
        Keys defining the normalisation / ranking pool.  Defaults to
        `group_cols` minus `candidate_col` (i.e. compare kernels within each
        kind x method cell -- the grand per-family semantics).  Pass e.g.
        ("system", "kind") to reproduce grand's cross-family "pooled" mode.
    metric_weights : dict, optional
        Defaults to `GRAND_WEIGHTS`.
    extra_metrics : DataFrame, optional
        Pre-computed per-candidate axes (e.g. agg_horizon_*, sc, vamp2) merged
        on the intersection of `group_cols`.  Anything still missing stays
        neutral, exactly as in the notebook.

    Returns
    -------
    (mode_agg_df, kernel_scores_df)
        `kernel_scores_df` carries `composite_score`, `admissible`,
        `constraint_violations`, `rank` (within pool) and `rank_overall`.
    """
    summary = summary.copy()
    group_cols = list(group_cols)

    if pool_cols is None:
        pool_cols = [c for c in group_cols if c != candidate_col]
    pool_cols = list(pool_cols)

    if larger_is_better is None:
        larger_is_better = set(LARGER_IS_BETTER)
    larger_is_better = set(larger_is_better)
    log_metrics = set(log_metrics or ())

    if metric_weights is None:
        metric_weights = dict(GRAND_WEIGHTS)

    if hard_constraints is _DEFAULT:
        hard_constraints = dict(HARD_CONSTRAINTS)

    # ---- 0: mode selection + mode weights ------------------------------------
    if selected_modes is not None:
        summary = summary[summary["eigenfunction_id"].isin(selected_modes)].copy()
    if summary.empty:
        raise ValueError("No rows remain in summary after filtering selected_modes.")

    if mode_weights is None:
        mode_weights = {m: 1.0 for m in sorted(summary["eigenfunction_id"].unique())}
    elif isinstance(mode_weights, Mapping):
        mode_weights = dict(mode_weights)
    else:
        mode_weights = {m: w for m, w in mode_weights}

    summary["mode_weight"] = summary["eigenfunction_id"].map(mode_weights).fillna(0.0)
    if (summary["mode_weight"] < 0).any():
        raise ValueError("mode_weights must be nonnegative.")

    # ----- 1: aggregate per-mode metrics --> per-candidate axes ---------
    def _wavg(g, col):
        w = g["mode_weight"].to_numpy(dtype=float)
        x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & (w > 0)
        if not ok.any():
            return np.nan
        return float(np.average(x[ok], weights=w[ok]))

    mode_agg_df = (
        summary.groupby(group_cols, as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_modes_used": int(g["eigenfunction_id"].nunique()),
                    "weight_sum": float(g["mode_weight"].sum()),
                    "agg_bias_mean": _wavg(g, "bias_mean"),
                    "agg_dist_mean": _wavg(g, "dist_mean"),
                    "agg_spurious_mean": _wavg(g, "spurious_mean"),
                    "agg_spurious_std": _wavg(g, "spurious_std"),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    trials_df = trials_df.copy() if trials_df is not None else None

    # ----- 2: aggregate the fixed per-trial axes if available ------
    if use_trial_metrics and trials_df is not None:
        tgroup = [c for c in group_cols if c in trials_df.columns]
        # map trial column -> (output name, aggfunc); include only if present
        trial_spec = {
            "spurious_ref_count": [("mean_spurious_ref_count", "mean")],
            "spurious_residual_count": [("mean_spurious_residual_count", "mean")],
            "spectral_gap": [
                ("mean_spectral_gap", "mean"),
                ("std_spectral_gap", "std"),
            ],
            "rank": [("mean_rank", "mean")],
        }
        named_agg = {
            out: (src, fn)
            for src, outs in trial_spec.items()
            if src in trials_df.columns
            for out, fn in outs
        }
        if tgroup and named_agg:
            trial_agg = trials_df.groupby(tgroup, as_index=False).agg(**named_agg)
            mode_agg_df = mode_agg_df.merge(trial_agg, on=tgroup, how="left")

    # ----- 3a: long horizon + SSL axes, optional -----------------
    # canonical output name -> accepted source aliases (raw or pre-aggregated).
    # Resolution precedence per axis:
    #   1. already on mode_agg_df (produced upstream)   -> keep
    #   2. per-trial column in trials_df                -> mean over trials
    #   3. per-mode column in summary                   -> mode-weighted average
    #   4. extra_metrics (explicit override)            -> merged last, below
    # Anything still unresolved stays absent -> neutral (z = 0) in the score.
    OPTIONAL_AXES = {
        "agg_horizon_instab": ("agg_horizon_instab", "horizon_instab"),
        "agg_horizon_sens": ("agg_horizon_sens", "horizon_sens"),
        "sc": ("sc",),
        "vamp2": ("vamp2",),
    }
    for out_name, aliases in OPTIONAL_AXES.items():
        if out_name in mode_agg_df.columns and mode_agg_df[out_name].notna().any():
            continue  # (1) already there

        # (2) per-trial source
        if use_trial_metrics and trials_df is not None:
            tgroup = [c for c in group_cols if c in trials_df.columns]
            tsrc = next((a for a in aliases if a in trials_df.columns), None)
            if tgroup and tsrc is not None:
                agg = (
                    trials_df.groupby(tgroup, as_index=False)[tsrc]
                    .mean()
                    .rename(columns={tsrc: out_name})
                )
                mode_agg_df = mode_agg_df.merge(agg, on=tgroup, how="left")
                continue

        # (3) per-mode source in summary (mode-weighted, like the other axes)
        ssrc = next((a for a in aliases if a in summary.columns), None)
        if ssrc is not None:
            wa = (
                summary.groupby(group_cols, as_index=False)
                .apply(
                    lambda g, c=ssrc: pd.Series({out_name: _wavg(g, c)}),
                    include_groups=False,
                )
                .reset_index(drop=True)
            )
            mode_agg_df = mode_agg_df.merge(wa, on=group_cols, how="left")

    # ----- 3b: explicit pre-computed per-candidate axes (optional override) -----
    if extra_metrics is not None:
        on = [
            c
            for c in group_cols
            if c in extra_metrics.columns and c in mode_agg_df.columns
        ]
        if on:
            add = [c for c in extra_metrics.columns if c not in on]
            # drop any placeholder columns we are about to override
            mode_agg_df = mode_agg_df.drop(
                columns=[c for c in add if c in mode_agg_df.columns]
            )
            mode_agg_df = mode_agg_df.merge(extra_metrics, on=on, how="left")

    # ----- 4: normalise (within pool) + weighted composite ------------
    df = mode_agg_df.copy().reset_index(drop=True)

    used_metrics = []
    for metric, weight in metric_weights.items():
        if weight == 0 or metric not in df.columns:
            continue
        if not pd.to_numeric(df[metric], errors="coerce").notna().any():
            continue  # axis entirely absent -> neutral (skip)
        df[f"{metric}_norm"] = _normalise_grouped(
            df,
            metric,
            pool_cols,
            method=normalise,
            larger_is_better=(metric in larger_is_better),
            log=(metric in log_metrics),
        )
        used_metrics.append(metric)

    score = np.zeros(len(df), dtype=float)
    for metric in used_metrics:
        score += metric_weights[metric] * df[f"{metric}_norm"].to_numpy(dtype=float)
    df[score_name] = score
    df["composite_score"] = score  # back-compatible alias
    df["used_metrics"] = ", ".join(used_metrics)

    # ----- 5: apply hard-constraints for admissibility ---------------------------
    admissible = np.ones(len(df), dtype=bool)
    viol = [[] for _ in range(len(df))]

    def _mark(mask, label):
        nonlocal admissible
        if mask is None:
            return
        mask = mask.fillna(False).to_numpy()
        admissible &= ~mask
        for i in np.where(mask)[0]:
            viol[i].append(label)

    hc = hard_constraints or {}
    _upper = {  # constraint key -> (column, label)
        "max_spurious_ref_count": ("mean_spurious_ref_count", "spurious_ref"),
        "max_spurious_residual_count": (
            "mean_spurious_residual_count",
            "spurious_residual",
        ),
        "max_dist_mean": ("agg_dist_mean", "distortion"),
        "max_bias_mean": ("agg_bias_mean", "bias"),
        "max_horizon_instab": ("agg_horizon_instab", "blowup"),  # |lambda| > 1
    }
    for key, (col, label) in _upper.items():
        if key in hc and col in df.columns:
            _mark(pd.to_numeric(df[col], errors="coerce") > hc[key], label)

    # Lower-bound (minimum) gates
    _lower = {  # constraint key -> (column, label)
        "min_koopman_gap": ("agg_koopman_gap", "gap"),
    }
    for key, (col, label) in _lower.items():
        if key in hc and col in df.columns:
            _mark(pd.to_numeric(df[col], errors="coerce") < hc[key], label)

    df["admissible"] = admissible
    df["constraint_violations"] = [",".join(v) for v in viol]

    # ----- 6: ranking (within pool; inadmissible sorted last) ---------
    df["_infeasible"] = (~df["admissible"]).astype(int)
    sort_keys = pool_cols + ["_infeasible", "composite_score"]
    srt = df.sort_values(sort_keys, kind="mergesort")

    if pool_cols:
        df["rank"] = (srt.groupby(pool_cols, dropna=False).cumcount() + 1).reindex(
            df.index
        )
    else:
        df["rank"] = pd.Series(np.arange(1, len(df) + 1), index=srt.index).reindex(
            df.index
        )

    srt_g = df.sort_values(["_infeasible", "composite_score"], kind="mergesort")
    df["rank_overall"] = pd.Series(
        np.arange(1, len(df) + 1), index=srt_g.index
    ).reindex(df.index)
    df = df.drop(columns="_infeasible")

    kernel_scores_df = (
        df.sort_values(pool_cols + ["rank"], kind="mergesort").reset_index(drop=True)
        if pool_cols
        else df.sort_values("rank_overall", kind="mergesort").reset_index(drop=True)
    )
    return mode_agg_df, kernel_scores_df


# ===========================================================================
# ----------------- sensitivity study around baseline weights ---------------
# ===========================================================================
def run_weight_sensitivity(
    summary,
    trials_df,
    *,
    base_weights=None,
    mode="scale",  # "scale": each weighted axis x `scales`; "drop": leave-one-out (weight -> 0)
    scales=(0.8, 1.0, 1.2),
    truth_col="true_eig_err",  # if present in `summary`, selection quality is scored against it
    selected_modes=(1, 2, 3),
    mode_weights=None,
    normalise="zscore",
    group_cols=("kernel", "kind", "method"),
    pool_cols=None,  # thesis FAM pool: ("family", "system", "kind_n", "method")
    hard_constraints=_DEFAULT,
    extra_metrics=None,
):
    """Sensitivity of the composite selection to the 7 WEIGHTED-axis weights
    only. Gates are thresholds, not weights, and are excluded from the sweep.

    Only keys with a non-zero `base_weights` entry are perturbed, so the gate
    metrics are excluded automatically.

    Function responsible for both reported weight ablations in §3.2.4:
    mode="scale"  perturbs each weighted axis in turn by every factor in `scales`
                  (baseline robustness, e.g. +/-20%).
    mode="drop"   sets each weighted axis to zero in turn (leave-one-out ablation:
                  does the axis earn its place?).

    Note on call-time: this function is best called with a truth-merged table.
    When `truth_col` is a column of `summary`, each configuration is scored per
    pool and reported as median within-pool Spearman and top-1-within-10% against
    it; `pools_moved` counts pools whose selected candidate differs from baseline.
    """
    from scipy.stats import spearmanr

    if base_weights is None:
        base_weights = dict(GRAND_WEIGHTS)
    if pool_cols is None:
        pool_cols = [c for c in group_cols if c != "kernel"]
    pool_cols = list(pool_cols)
    axes = [m for m, w in base_weights.items() if w]  # weighted axes only

    def _score(weights):
        _, scores = composite_score(
            summary,
            trials_df=trials_df,
            group_cols=group_cols,
            pool_cols=pool_cols,
            selected_modes=list(selected_modes),
            mode_weights=mode_weights,
            normalise=normalise,
            metric_weights=weights,
            hard_constraints=hard_constraints,
            extra_metrics=extra_metrics,
        )
        return scores

    def _pick(scores):  # per-pool selected candidate
        adm = scores[scores["admissible"]] if "admissible" in scores else scores
        adm = adm if len(adm) else scores
        return {
            k: g.loc[g["composite_score"].idxmin()].name
            for k, g in adm.groupby(pool_cols)
        }

    def _quality(scores):
        if truth_col not in scores.columns:
            return (np.nan, np.nan, 0)
        S, T = [], []
        for _, g in scores.groupby(pool_cols):
            g = g.dropna(subset=["composite_score", truth_col])
            if len(g) < 3 or g[truth_col].nunique() < 2:
                continue
            best = g[truth_col].min()
            pk = g.loc[g["composite_score"].idxmin()]
            S.append(spearmanr(g["composite_score"], g[truth_col]).statistic)
            T.append(
                1.0
                if (pk[truth_col] - best < 1e-9)
                or (best > 0 and (pk[truth_col] - best) / best < 0.10)
                else 0.0
            )
        return (
            (float(np.median(S)), float(np.mean(T)), len(S))
            if S
            else (np.nan, np.nan, 0)
        )

    base_scores = _score(base_weights)
    base_pick = _pick(base_scores)
    b_sp, b_t1, b_n = _quality(base_scores)

    if mode == "scale":
        configs = [
            (m, s, {**base_weights, m: base_weights[m] * s})
            for m in axes
            for s in scales
        ]
    elif mode == "drop":
        configs = [(m, 0.0, {**base_weights, m: 0.0}) for m in axes]
    else:
        raise ValueError("mode must be 'scale' or 'drop'")

    rows = [
        {
            "axis": "(baseline)",
            "scale": 1.0,
            "median_spearman": b_sp,
            "top1": b_t1,
            "n_pools": b_n,
            "pools_moved": 0,
        }
    ]
    for m, s, w in configs:
        sc = _score(w)
        sp, t1, n = _quality(sc)
        pk = _pick(sc)
        moved = sum(1 for k in base_pick if pk.get(k) != base_pick[k])
        rows.append(
            {
                "axis": m,
                "scale": s,
                "median_spearman": sp,
                "top1": t1,
                "n_pools": n,
                "pools_moved": moved,
            }
        )

    return pd.DataFrame(rows)
