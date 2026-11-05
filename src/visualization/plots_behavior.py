"""
Behavioural plots for the empirical (Track B) analyses: psychometric / chronometric curves,
Conditional Bias Functions and mouse heterogeneity. Joshua Koilpillai.

All functions take any §3.1 trial DataFrame (or the tables produced by `src.empirical`),
return the matplotlib Figure, and optionally save it. They never change global matplotlib
settings (a local `rc_context` is used) and are backend-agnostic.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..empirical._common import biased_trials, require_columns
from ..empirical.cbf import cbf_by_subject, cohort_cbf
from ..empirical.psychometrics import (
    across_subject,
    chronometric_table,
    fit_psychometric_by,
    psychometric_function,
    psychometric_table,
)

RIGHT_COLOR = "#2b83ba"  # right-biased block (80/20), matches plots_recovery
LEFT_COLOR = "#d7191c"   # left-biased block (20/80)
_RC = {
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0, "axes.grid": True,
    "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.alpha": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}
_PRIORS = ((1.0, RIGHT_COLOR, "Right block (80/20)"), (-1.0, LEFT_COLOR, "Left block (20/80)"))


def _finish(fig: plt.Figure, save_path: Optional[str], dpi: int = 200) -> plt.Figure:
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _grid(n: int, ncols: int, cell=(3.0, 2.6)):
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows),
                             squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def plot_psychometric_chronometric_cohort(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Cohort psychometric and chronometric functions",
) -> plt.Figure:
    """
    Cohort psychometric (P(Right)) and chronometric (mean RT) curves for each block prior.
    Thin lines = individual mice; markers with bars = mean ± SEM across mice.
    """
    require_columns(df)
    psy = psychometric_table(df)
    chrono = chronometric_table(df)
    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for prior, color, label in _PRIORS:
            for tab, ax, col in ((psy, ax1, "p_right"), (chrono, ax2, "mean_rt")):
                t = tab[tab["prior"] == prior]
                for _, g in t.groupby("subject_id", observed=True):
                    ax.plot(g["contrast"], g[col], color=color, alpha=0.2, lw=1)
                m = across_subject(t, col, ["contrast"])
                ax.errorbar(m["contrast"], m["mean"], yerr=m["sem"], color=color, marker="o", ms=6,
                            lw=2, capsize=3, label=label)
        ax1.axhline(0.5, color="gray", ls=":", lw=1)
        ax1.axvline(0, color="gray", ls=":", lw=1)
        ax1.set(xlabel="Signed contrast (+ = right)", ylabel="P(choose Right)", title="Psychometric function")
        ax2.set(xlabel="Signed contrast (+ = right)", ylabel="Mean reaction time (s)", title="Chronometric function")
        ax1.legend(fontsize=9, frameon=False)
        fig.suptitle(title, fontweight="bold")
    return _finish(fig, save_path)


def plot_psychometric_by_mouse(
    df: pd.DataFrame,
    fits: Optional[pd.DataFrame] = None,
    ncols: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Per-mouse psychometric panels: data with 95% Wilson bars and (optional) fitted curves."""
    require_columns(df)
    if fits is None:
        fits = fit_psychometric_by(df)
    psy = psychometric_table(df)
    subjects = sorted(psy["subject_id"].unique())
    xs = np.linspace(-1, 1, 400)
    with plt.rc_context(_RC):
        fig, axes = _grid(len(subjects), ncols)
        for ax, subj in zip(axes, subjects):
            for prior, color, _ in _PRIORS:
                t = psy[(psy["subject_id"] == subj) & (psy["prior"] == prior)]
                ax.errorbar(t["contrast"], t["p_right"],
                            yerr=[np.clip(t["p_right"] - t["ci_lo"], 0, None),
                                  np.clip(t["ci_hi"] - t["p_right"], 0, None)],
                            fmt="o", color=color, ms=4, lw=1, capsize=0)
                f = fits[(fits["subject_id"] == subj) & (fits["prior"] == prior)]
                if len(f) and np.isfinite(f["mu"].iloc[0]):
                    r = f.iloc[0]
                    ax.plot(xs, psychometric_function(xs, r["mu"], r["sigma"], r["gamma"], r["lam"]),
                            color=color, lw=1.5)
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
            ax.set_title(str(subj), fontsize=9)
        for ax in axes[::ncols]:
            ax.set_ylabel("P(Right)")
        fig.supxlabel("Signed contrast")
    return _finish(fig, save_path)


def plot_chronometric_by_mouse(
    df: pd.DataFrame, ncols: int = 5, save_path: Optional[str] = None
) -> plt.Figure:
    """Per-mouse chronometric panels: mean RT ± SEM vs signed contrast for each block prior."""
    require_columns(df)
    chrono = chronometric_table(df)
    subjects = sorted(chrono["subject_id"].unique())
    with plt.rc_context(_RC):
        fig, axes = _grid(len(subjects), ncols)
        for ax, subj in zip(axes, subjects):
            for prior, color, _ in _PRIORS:
                t = chrono[(chrono["subject_id"] == subj) & (chrono["prior"] == prior)]
                ax.errorbar(t["contrast"], t["mean_rt"], yerr=t["sem_rt"], marker="o", ms=4,
                            color=color, lw=1.2, capsize=0)
            ax.set_title(str(subj), fontsize=9)
        for ax in axes[::ncols]:
            ax.set_ylabel("Mean RT (s)")
        fig.supxlabel("Signed contrast")
    return _finish(fig, save_path)


def plot_cbf_cohort(
    curves: pd.DataFrame,
    overlay: Optional[Dict[str, Sequence[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Empirical Conditional Bias Function",
) -> plt.Figure:
    """
    Cohort CBF over RT quintiles: thin lines = mice, bold = mean ± SEM across mice.

    `overlay` maps a label to per-bin CBF values (length n_bins) for model predictions, e.g.
    {'M1': cbf_array, ...} in Week 3. Bins are plotted by quantile index so models with
    different RT scales can be compared directly.
    """
    coh = cohort_cbf(curves)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for _, g in curves.groupby("subject_id", observed=True):
            ax.plot(g["bin"] + 1, g["cbf"], color="#888888", alpha=0.35, lw=1)
        ax.errorbar(coh["bin"] + 1, coh["cbf"], yerr=coh["sem"], color="#111111", marker="D", ms=7,
                    lw=2.4, capsize=3, label=f"Mice (mean ± SEM, n={int(coh['n_subjects'].max())})")
        for (label, vals), color in zip((overlay or {}).items(), ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")):
            ax.plot(np.arange(1, len(vals) + 1), vals, "--", marker="o", color=color, lw=1.8, label=label)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set(xlabel="RT quantile bin (1 = fastest)", ylabel="Prior bias ΔP(Right | prior)",
               title=title, xticks=range(1, len(coh) + 1))
        ax.legend(frameon=False, fontsize=9)
    return _finish(fig, save_path)


def plot_cbf_by_mouse(
    curves: pd.DataFrame,
    slopes: Optional[pd.DataFrame] = None,
    ncols: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Per-mouse CBF panels with bootstrap bands; panel titles carry the slope-based shape label."""
    subjects = sorted(curves["subject_id"].unique())
    shape_color = {"falling": "#1f77b4", "flat": "#2ca02c", "rising": "#d62728", "undetermined": "#888888"}
    with plt.rc_context(_RC):
        fig, axes = _grid(len(subjects), ncols)
        for ax, subj in zip(axes, subjects):
            g = curves[curves["subject_id"] == subj]
            shape = "undetermined"
            if slopes is not None and (slopes["subject_id"] == subj).any():
                shape = slopes.loc[slopes["subject_id"] == subj, "shape"].iloc[0]
            c = shape_color.get(shape, "#444444")
            ax.fill_between(g["bin"] + 1, g["cbf_lo"], g["cbf_hi"], color=c, alpha=0.2)
            ax.plot(g["bin"] + 1, g["cbf"], marker="o", color=c, lw=1.8)
            ax.axhline(0, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{subj} ({shape})", fontsize=9)
            ax.set_xticks(range(1, int(g["bin"].max()) + 2))
        for ax in axes[::ncols]:
            ax.set_ylabel("ΔP(Right)")
        fig.supxlabel("RT quantile bin (1 = fastest)")
    return _finish(fig, save_path)


def plot_rt_distributions(
    df: pd.DataFrame, save_path: Optional[str] = None, rt_max: float = 2.5
) -> plt.Figure:
    """Empirical CDF of RT for each mouse (left) and RT density on a log axis (right)."""
    require_columns(df)
    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        bins = np.logspace(np.log10(0.1), np.log10(rt_max), 40)
        for subj, g in df.groupby("subject_id", observed=True):
            rt = np.sort(g["rt"].to_numpy())
            ax1.plot(rt, np.arange(1, len(rt) + 1) / len(rt), lw=1.3, label=str(subj))
            ax2.hist(rt, bins=bins, density=True, histtype="step", lw=1.3)
        ax1.set(xscale="log", xlabel="RT (s)", ylabel="Cumulative probability", title="RT CDF per mouse")
        ax2.set(xscale="log", xlabel="RT (s)", ylabel="Density", title="RT distribution (log-spaced bins)")
        if df["subject_id"].nunique() <= 12:
            ax1.legend(fontsize=7, frameon=False, ncol=2)
    return _finish(fig, save_path)


def plot_heterogeneity(profile: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Dot plots of key behavioural metrics across mice (one panel per metric, sorted by value)."""
    panels = [
        ("sigma", "Psychometric slope scale σ"), ("gamma", "Left-stimulus lapse γ (P(R) at far left)"), ("lam", "Right-stimulus lapse λ (P(L) at far right)"),
        ("prior_shift", "Prior shift (contrast units)"), ("prior_effect_c0", "ΔP(Right) at 0% contrast"),
        ("history_effect_c0", "History effect at 0% contrast"), ("rt_median", "Median RT (s)"),
        ("frac_slow", "Fraction RT > 1 s"), ("cbf_slope", "CBF slope per RT bin"),
        ("gamma_block_diff", "Prior effect on γ (right − left block)"),
        ("lam_block_diff", "Prior effect on λ (right − left block)"),
        ("repeat_rate", "Choice repeat rate"),
    ]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        for ax, (col, label) in zip(axes.ravel(), panels):
            p = profile.sort_values(col).reset_index(drop=True)
            ax.scatter(p[col], range(len(p)), color="#222222", zorder=3)
            if col == "prior_shift" and "prior_shift_lo" in p:
                ax.hlines(range(len(p)), p["prior_shift_lo"], p["prior_shift_hi"], color="#555555", lw=1.5)
            if col == "cbf_slope" and "cbf_slope_lo" in p:
                ax.hlines(range(len(p)), p["cbf_slope_lo"], p["cbf_slope_hi"], color="#555555", lw=1.5)
                ax.axvline(0, color="gray", ls=":", lw=1)
            ax.set_yticks(range(len(p)))
            ax.set_yticklabels(p["subject_id"], fontsize=8)
            ax.set_title(label, fontsize=10)
        fig.suptitle("Individual-mouse heterogeneity", fontweight="bold")
    return _finish(fig, save_path)


def plot_cbf_sensitivity(
    df: pd.DataFrame,
    settings: Dict[str, dict],
    n_boot: int = 200,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Cohort CBF under alternative analysis settings, e.g.
    {'0% contrast only': {'max_abs_contrast': 0.0}, '|c| <= 0.125': {...}, 'all contrasts': {...}}.
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for (label, kwargs), color in zip(settings.items(), ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")):
            curves, _ = cbf_by_subject(biased_trials(df), n_boot=n_boot, **kwargs)
            coh = cohort_cbf(curves)
            ax.errorbar(coh["bin"] + 1, coh["cbf"], yerr=coh["sem"], marker="o", lw=2, capsize=3,
                        color=color, label=label)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set(xlabel="RT quantile bin (1 = fastest)", ylabel="Prior bias ΔP(Right | prior)",
               title="CBF sensitivity to contrast selection")
        ax.legend(frameon=False, fontsize=9)
    return _finish(fig, save_path)
