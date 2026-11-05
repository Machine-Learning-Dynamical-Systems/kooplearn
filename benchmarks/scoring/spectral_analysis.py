# 0091/benchmarks/scoring/spectral_analysis.py
# ===========================================================================
# Updated `analyse_spectrum` that also emits the extended grand-score axes:
#   agg_horizon_instab, agg_horizon_sens   (from the est_eig columns of modes_df)
#   sc, vamp2                              (mean over trials, if present)
#
# These four are CANDIDATE-level (one value per kernel x kind x method), not
# per-mode, so they are collected into a separate `axes_df` rather than forced
# into the per-mode `summary`.  `axes_df` is exactly the frame that the grand
# scorer wants: kernel_spectral_score(summary, trials_df, extra_metrics=axes_df).
#
# Everything is optional and degrades gracefully -- runs that do not compute
# est_eig / sc / vamp2 behave exactly like the original analyse_spectrum, just
# with an empty axes table.  Original benchmarks/kernel_spectrum.py is untouched.
# ===========================================================================

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics.representation import MODE_W, H, horizon_terms


def analyse_spectrum(
    modes_records,
    trials_records,
    out_prefix,
    *,
    compute_extra_axes=True,
    horizon_H=H,
    horizon_mode_weights=MODE_W,
):
    modes_df = pd.DataFrame(modes_records).copy()
    trials_df = pd.DataFrame(trials_records).copy()

    if "spectral_gap" not in modes_df.columns:
        raise ValueError(
            f"modes_df is missing 'spectral_gap'. Columns: {modes_df.columns.tolist()}"
        )

    summary = modes_df.groupby(
        ["kernel", "kind", "method", "eigenfunction_id"], as_index=False
    ).agg(
        n=("spectral_bias", "size"),
        bias_mean=("spectral_bias", "mean"),
        bias_std=("spectral_bias", "std"),
        dist_mean=("metric_distortion", "mean"),
        trunc_mean=("truncation", "mean"),
        spurious_mean=("residual_spurious_score", "mean"),
        spurious_std=("residual_spurious_score", "std"),
    )

    rows = []
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        gg = g[["spectral_bias", "spectral_gap"]].dropna()
        corr = gg["spectral_bias"].corr(gg["spectral_gap"]) if len(gg) > 1 else np.nan
        rows.append(
            {
                "kernel": kernel,
                "kind": kind,
                "method": method,
                "bias_gap_corr": corr,
            }
        )
    corr_df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Extended grand-score axes (candidate-level, all optional)
    # ------------------------------------------------------------
    axes_df = _extended_axes(
        modes_df,
        trials_df,
        compute_extra_axes=compute_extra_axes,
        horizon_H=horizon_H,
        horizon_mode_weights=horizon_mode_weights,
    )

    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    modes_df.to_csv(f"{out_prefix}_metrics.csv", index=False)
    trials_df.to_csv(f"{out_prefix}_trials.csv", index=False)
    corr_df.to_csv(f"{out_prefix}_corr.csv", index=False)
    axes_df.to_csv(f"{out_prefix}_axes.csv", index=False)

    # ------------------------------------------------------------
    # Scatter diagnostics with compact, deduplicated legends
    # ------------------------------------------------------------
    def _short_label(x, max_chars=24):
        s = str(x)
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*\s*=\s*[^,)]+)", s)
        if m:
            return m.group(1).replace(" ", "")
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

    def _build_group_label(kernel, kind, method, include_kernel=False):
        """
        Keep legend labels compact.
        By default, legend shows only kind + method.
        Set include_kernel=True only if the number of groups is small.
        """
        if include_kernel:
            return f"{_short_label(kernel)}, {kind} / {method}"
        return f"{kind} / {method}"

    def _dedup_legend(
        ax, title=None, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8
    ):
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l != "_nolegend_" and l not in seen:
                seen[l] = h
        if seen:
            ax.legend(
                seen.values(),
                seen.keys(),
                frameon=False,
                fontsize=fontsize,
                title=title,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                borderaxespad=0.0,
            )

    # -----------------------------
    # Figure 1: spectral bias vs spectral gap
    # -----------------------------
    fig1, ax = plt.subplots(figsize=(8.5, 4.8))
    # Set this to True only when there are very few groups
    include_kernel_in_legend = True
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        g = g[np.isfinite(g["spectral_bias"]) & np.isfinite(g["spectral_gap"])].copy()
        if g.empty:
            continue
        label = _build_group_label(
            kernel=kernel,
            kind=kind,
            method=method,
            include_kernel=include_kernel_in_legend,
        )
        ax.scatter(
            g["spectral_bias"],
            g["spectral_gap"],
            s=20,
            alpha=0.7,
            label=label,
        )
    ax.set_xlabel("Spectral bias")
    ax.set_ylabel("Spectral gap")
    ax.set_title("Spectral bias vs Spectral gap")
    ax.grid(alpha=0.25)
    _dedup_legend(
        ax,
        title="Condition / method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_gap_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # -----------------------------
    # Figure 2: spectral bias vs residual spurious score
    # -----------------------------
    fig2, ax = plt.subplots(figsize=(8.5, 4.8))
    for (kernel, kind, method), g in modes_df.groupby(["kernel", "kind", "method"]):
        g = g[
            np.isfinite(g["spectral_bias"]) & np.isfinite(g["residual_spurious_score"])
        ].copy()
        if g.empty:
            continue
        label = _build_group_label(
            kernel=kernel,
            kind=kind,
            method=method,
            include_kernel=include_kernel_in_legend,
        )
        ax.scatter(
            g["spectral_bias"],
            g["residual_spurious_score"],
            s=20,
            alpha=0.7,
            label=label,
        )
    ax.set_xlabel("Spectral bias")
    ax.set_ylabel("Residual spurious score")
    ax.set_title("Spectral bias vs Residual spurious score")
    ax.grid(alpha=0.25)
    _dedup_legend(
        ax,
        title="Condition / method",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_spurious_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # NOTE: return arity extended by one -- `axes_df` sits next to summary/corr_df.
    return modes_df, trials_df, summary, corr_df, axes_df, fig1, fig2


def _extended_axes(
    modes_df,
    trials_df,
    *,
    compute_extra_axes=True,
    horizon_H=H,
    horizon_mode_weights=MODE_W,
):
    """Candidate-level table of the extended axes; empty (keys only) if none apply.

    - horizon terms: computed from est_eig_real/est_eig_imag in `modes_df`
    - sc / vamp2:    averaged over trials from `trials_df` (whichever exist)

    Returned columns: kernel, kind, method [, agg_horizon_instab,
    agg_horizon_sens, sc, vamp2].  Feed directly to
    kernel_spectral_score(..., extra_metrics=axes_df).
    """
    keys = ["kernel", "kind", "method"]
    base = (
        modes_df[keys].drop_duplicates().reset_index(drop=True)
        if set(keys).issubset(modes_df.columns)
        else pd.DataFrame(columns=keys)
    )
    if not compute_extra_axes or base.empty:
        return base

    parts = [base.set_index(keys)]

    # --- long-horizon terms (need per-trial est_eig columns) ---------------
    if {"est_eig_real", "est_eig_imag"}.issubset(modes_df.columns):
        hz = horizon_terms(
            modes_df, H=horizon_H, mode_weights=horizon_mode_weights, group_cols=keys
        )
        if len(hz):
            parts.append(hz.set_index(keys))

    # --- SSL criteria (mean over trials / data draws) ----------------------
    ssl_cols = [
        c
        for c in ("sc", "vamp2", "vamp2_ridge", "sc_star", "hs_gap")
        if c in trials_df.columns
    ]
    tkeys = [c for c in keys if c in trials_df.columns]
    if ssl_cols and tkeys:
        ssl = trials_df.groupby(tkeys, as_index=False)[ssl_cols].mean()
        parts.append(ssl.set_index(tkeys))

    axes_df = pd.concat(parts, axis=1).reset_index()
    return axes_df
