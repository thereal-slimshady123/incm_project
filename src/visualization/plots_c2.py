"""
Figures for the C2 (boundary dynamics / urgency) analysis. Inputs are the tables produced by
`src.optimization.c2`; every function returns the figure and optionally saves it.
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots_behavior import _RC, _finish
from .plots_c1 import EMP_COLOR, MODEL_COLORS

BOUND_COLORS = {"fixed": "#3b6ea5", "weibull": "#e17c05"}
SOURCE_COLORS = {"data": EMP_COLOR, **BOUND_COLORS}
SOURCE_LABELS = {"data": "Mice", "fixed": "Fixed bound", "weibull": "Weibull collapsing bound"}


def plot_bound_comparison(
    tables: Dict[str, pd.DataFrame],
    thresholds: Optional[Dict[str, float]] = None,
    metric: str = "bic",
    save_path: Optional[str] = None,
    title: str = "C2: collapsing vs fixed bound, per mouse",
) -> plt.Figure:
    """
    One panel per variant (e.g. '2.5 s cutoff', '5.0 s cutoff'; each value a `c2.bound_table`): delta-BIC of the Weibull
    fit relative to the fixed-bound fit per mouse (negative = the collapsing bound is better). Grey band = inconclusive
    range, horizontal line = cohort mean.
    """
    col = f"d{metric}_weibull_vs_fixed"
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(tables), figsize=(5.6 * len(tables), 4.4), squeeze=False)
        for ax, (name, tab) in zip(axes[0], tables.items()):
            t = tab.sort_values("subject_id").reset_index(drop=True)
            x = np.arange(len(t))
            thr = (thresholds or {}).get(name, 10.0)
            ax.axhspan(-thr, thr, color="#dddddd", alpha=0.7, lw=0, label=f"inconclusive band (|Δ| < {thr:.0f})")
            ax.axhline(0, color="#555555", lw=1)
            ax.axhline(t[col].mean(), color=BOUND_COLORS["weibull"], ls="--", lw=1.4, label=f"mean {t[col].mean():+.0f}")
            ax.scatter(x, t[col], s=60, color=BOUND_COLORS["weibull"], zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(t["subject_id"], rotation=40, ha="right", fontsize=9)
            ax.set(ylabel=f"Δ{metric.upper()}  (Weibull − fixed; negative = collapsing better)", title=name)
            lim = max(float(np.nanmax(np.abs(t[col]))), 30.0)
            ax.set_ylim(-1.5 * lim, 1.5 * lim)          # zoom on the points; the decisive band may extend beyond the axes
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_fitted_bounds(
    curves: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "C2: fitted Weibull bound a(t)/a",
    rt_marks: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Fitted collapsing bounds of all mice (thin) and their cohort median (bold); `rt_marks` draws labelled vertical lines."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for _, g in curves.groupby("subject_id"):
            ax.plot(g["t"], g["fraction"], color=BOUND_COLORS["weibull"], alpha=0.35, lw=1.2)
        med = curves.groupby("t")["fraction"].median()
        ax.plot(med.index, med.values, color=BOUND_COLORS["weibull"], lw=2.8, label="cohort median")
        ax.axhline(1.0, color=BOUND_COLORS["fixed"], ls="--", lw=1.5, label="fixed bound")
        for label, x in (rt_marks or {}).items():
            ax.axvline(x, color="#999999", ls=":", lw=1)
            ax.text(x, 0.02, f" {label}", fontsize=8, color="#555555", rotation=90, va="bottom")
        ax.set(xlabel="Decision time t (s)", ylabel="Bound height a(t) / a", ylim=(0, 1.05), title=title)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _finish(fig, save_path)


def plot_invariance(
    inv: pd.DataFrame,
    metric: str = "bic",
    save_path: Optional[str] = None,
    title: str = "C2: is the C1 comparison the same under a collapsing bound?",
) -> plt.Figure:
    """
    Per model pair (M1 vs M2, M3 vs M2): each mouse's delta-BIC under the fixed bound (x) against the Weibull bound (y).
    Points on the diagonal = unchanged by the bound; the same sign on both axes = the same preferred model.
    """
    pairs = [("M1", "M2"), ("M3", "M2")]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 4.8))
        for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
            name = f"d{metric}_{a}_vs_{b}"
            x, y = inv[f"{name}_fixed"], inv[f"{name}_weibull"]
            lim = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.1
            ax.axhline(0, color="#aaaaaa", lw=0.8)
            ax.axvline(0, color="#aaaaaa", lw=0.8)
            ax.plot([-lim, lim], [-lim, lim], color="gray", ls=":", lw=1)
            ax.scatter(x, y, s=55, color=MODEL_COLORS[a], zorder=3)
            for s, xi, yi in zip(inv["subject_id"], x, y):
                ax.annotate(s, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=f"{a} vs {b} (positive = {a} worse)",
                   xlabel=f"Δ{metric.upper()} with fixed bound", ylabel=f"Δ{metric.upper()} with Weibull bound")
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def _mean_sem(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    g = df.groupby(x)[y]
    return pd.DataFrame({"mean": g.mean(), "sem": g.std(ddof=1) / np.sqrt(g.count())}).reset_index()


def plot_tail_diagnostics(
    tt: Dict[str, pd.DataFrame],
    sources: Sequence[str] = ("data", "fixed", "weibull"),
    save_path: Optional[str] = None,
    title: str = "C2: RT tail, mice vs fitted fixed and collapsing bounds",
    exclude: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Left: survival curve P(RT > t) (log axis). Middle: hazard in 0.5 s windows (rising hazard = urgency, falling = a mixture
    of slow processes). Right: accuracy for |contrast| >= 25% in fixed RT windows (urgency predicts decay toward chance).
    Cohort mean +- SEM over mice for the data, cohort mean for the model predictions.
    """
    tt = {k: (v[~v["subject_id"].isin(exclude)] if exclude else v) for k, v in tt.items()}
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for src in sources:
            kw = dict(color=SOURCE_COLORS[src], label=SOURCE_LABELS[src], lw={"data": 2.2, "fixed": 4.5, "weibull": 1.6}[src],
                      ls="-" if src != "weibull" else "--", marker="D" if src == "data" else "o", ms=4,
                      alpha=0.55 if src == "fixed" else 1.0)     # fixed is drawn wide underneath: the two coincide when lam hits its ceiling
            s = _mean_sem(tt["survival"][tt["survival"]["source"] == src], "t", "survival")
            axes[0].errorbar(s["t"], s["mean"], yerr=s["sem"] if src == "data" else None, capsize=2, **kw)
            h = tt["hazard"][tt["hazard"]["source"] == src].copy()
            h["t_mid"] = (h["t_lo"] + h["t_hi"]) / 2
            hm = _mean_sem(h, "t_mid", "hazard")
            axes[1].errorbar(hm["t_mid"], hm["mean"], yerr=hm["sem"] if src == "data" else None, capsize=2, **kw)
            a = tt["accuracy"][tt["accuracy"]["source"] == src].copy()
            a["t_mid"] = (a["t_lo"] + a["t_hi"]) / 2
            am = _mean_sem(a, "t_mid", "accuracy")
            axes[2].errorbar(am["t_mid"], am["mean"], yerr=am["sem"] if src == "data" else None, capsize=2, **kw)
        axes[0].set(yscale="log", xlabel="RT t (s)", ylabel="P(RT > t)", title="Survival")
        axes[1].set(xlabel="RT (s), window midpoint", ylabel="Hazard (1/s)", title="Hazard")
        axes[2].set(xlabel="RT (s), window midpoint", ylabel="Accuracy, |contrast| ≥ 25%", title="Accuracy by RT", ylim=(0.4, 1.0))
        axes[2].axhline(0.5, color="gray", ls=":", lw=1)
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_tail_by_mouse(
    tt: Dict[str, pd.DataFrame],
    ncols: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Per-mouse survival curves (log axis): data against the fixed and collapsing predictions."""
    from .plots_behavior import _grid
    subjects = sorted(tt["survival"]["subject_id"].unique())
    with plt.rc_context(_RC):
        fig, axes = _grid(len(subjects), ncols, cell=(3.2, 2.8))
        for ax, s in zip(axes, subjects):
            for src in ("data", "fixed", "weibull"):
                g = tt["survival"]
                g = g[(g["subject_id"] == s) & (g["source"] == src)].sort_values("t")
                if len(g):
                    ax.plot(g["t"], g["survival"].clip(lower=1e-4), color=SOURCE_COLORS[src], lw=2 if src == "data" else 1.4,
                            ls="-" if src == "data" else "--", label=SOURCE_LABELS[src])
            ax.set(yscale="log", title=s)
            ax.set_title(s, fontsize=9)
        axes[0].legend(frameon=False, fontsize=7)
        fig.supxlabel("RT t (s)")
        fig.supylabel("P(RT > t)")
    return _finish(fig, save_path)
