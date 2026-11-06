"""
Figures for the C1 (prior mechanism) adjudication: ΔBIC per mouse and model-vs-data CBF overlays.
Inputs are the tables produced by `src.optimization.c1`.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..optimization.c1 import MODEL_LABELS, MODELS
from .plots_behavior import _RC, _finish, _grid

MODEL_COLORS = {"M1": "#1f77b4", "M2": "#2ca02c", "M3": "#d62728", "JOINT": "#9467bd"}   # same order as plot_cbf_cohort overlays
EMP_COLOR = "#111111"


def plot_delta_bic(
    table: pd.DataFrame,
    threshold: float,
    metric: str = "bic",
    save_path: Optional[str] = None,
    title: str = "C1: model comparison per mouse",
) -> plt.Figure:
    """
    Left: ΔBIC of M1 and M3 relative to the M2 baseline per mouse (below 0 = better than M2); the grey
    band is the |ΔBIC| range called inconclusive. Right: number of mice won by each model (all, and
    decisive only).
    """
    t = table.sort_values("subject_id").reset_index(drop=True)
    x = np.arange(len(t))
    with plt.rc_context(_RC):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2.6, 1]})
        ax.axhspan(-threshold, threshold, color="#dddddd", alpha=0.7, lw=0, label=f"inconclusive (|Δ| < {threshold:.0f})")
        ax.axhline(0, color="#555555", lw=1)
        for off, m in ((-0.12, "M1"), (0.12, "M3")):
            col = f"d{metric}_{m}_vs_M2"
            ax.scatter(x + off, t[col], s=60, color=MODEL_COLORS[m], zorder=3, label=f"{MODEL_LABELS[m]} − M2")
        ax.set_xticks(x)
        ax.set_xticklabels(t["subject_id"], rotation=40, ha="right", fontsize=9)
        ax.set(ylabel=f"Δ{metric.upper()} vs M2 (negative = better than M2)", title=title)
        lim = max(1.1 * threshold, float(np.nanmax(np.abs(t[[f"d{metric}_M1_vs_M2", f"d{metric}_M3_vs_M2"]].to_numpy()))) * 1.05)
        if lim > 8 * threshold:
            ax.set_yscale("symlog", linthresh=threshold)
        ax.legend(frameon=False, fontsize=8, loc="best")

        n_best = [(t["winner"] == m).sum() for m in MODELS]
        n_dec = [(t["winner_decisive"] == m).sum() for m in MODELS]
        w = 0.38
        xs = np.arange(len(MODELS))
        ax2.bar(xs - w / 2, n_best, w, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.45, label="lowest BIC")
        ax2.bar(xs + w / 2, n_dec, w, color=[MODEL_COLORS[m] for m in MODELS], label="decisive")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(MODELS)
        ax2.set(ylabel="Mice", title="Winning model", ylim=(0, len(t)))
        ax2.legend(frameon=False, fontsize=8)
    return _finish(fig, save_path)


def _cohort_mean(df: pd.DataFrame, value: str = "cbf") -> pd.DataFrame:
    g = df.groupby("bin")[value]
    return pd.DataFrame({"mean": g.mean(), "sem": g.std(ddof=1) / np.sqrt(g.count())}).reset_index()


def plot_cbf_overlay_cohort(
    emp: pd.DataFrame,
    pred: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "C1: model-predicted vs empirical CBF",
    exclude: Optional[Sequence[str]] = None,
    models: Sequence[str] = MODELS,
) -> plt.Figure:
    """
    Four panels over RT quantile bins: all three model predictions with the empirical cohort CBF, then
    one panel per model (thin lines = individual mice's predictions, bold = cohort mean). Predictions
    are scored on each mouse's own empirical RT bins, so bin k is the same RT range for data and model.
    """
    if exclude:
        emp, pred = emp[~emp["subject_id"].isin(exclude)], pred[~pred["subject_id"].isin(exclude)]
    e = _cohort_mean(emp)
    n_subj = emp["subject_id"].nunique()
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.0 * (len(models) + 1), 4.0), sharey=True)
        ax = axes[0]
        ax.errorbar(e["bin"] + 1, e["mean"], yerr=e["sem"], color=EMP_COLOR, marker="D", ms=7, lw=2.4, capsize=3,
                    label=f"Mice (n={n_subj})", zorder=5)
        for m in models:
            pm = _cohort_mean(pred[pred["model"] == m])
            ax.plot(pm["bin"] + 1, pm["mean"], "--", marker="o", color=MODEL_COLORS[m], lw=1.8, label=MODEL_LABELS[m])
        ax.set_title("Cohort mean ± SEM")
        for ax, m in zip(axes[1:], models):
            for _, g in pred[pred["model"] == m].groupby("subject_id"):
                ax.plot(g["bin"] + 1, g["cbf"], color=MODEL_COLORS[m], alpha=0.25, lw=1)
            pm = _cohort_mean(pred[pred["model"] == m])
            ax.plot(pm["bin"] + 1, pm["mean"], color=MODEL_COLORS[m], marker="o", lw=2.4, label=f"{m} (mean)")
            ax.errorbar(e["bin"] + 1, e["mean"], yerr=e["sem"], color=EMP_COLOR, marker="D", ms=5, lw=1.8, capsize=2,
                        label="Mice (mean)", zorder=5)
            ax.set_title(MODEL_LABELS[m])
        for ax in axes:
            ax.axhline(0, color="gray", ls=":", lw=1)
            ax.set_xticks(range(1, int(emp["bin"].max()) + 2))
            ax.set_xlabel("RT quantile bin (1 = fastest)")
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Prior bias ΔP(Right | prior)")
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_cbf_overlay_by_mouse(
    emp: pd.DataFrame,
    pred: pd.DataFrame,
    table: Optional[pd.DataFrame] = None,
    ncols: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Per-mouse CBF: empirical (black, 95% band) against the three fitted models; titles show the BIC winner."""
    subjects = sorted(emp["subject_id"].unique())
    winners = {} if table is None else dict(zip(table["subject_id"], zip(table["winner"], table["evidence"])))
    with plt.rc_context(_RC):
        fig, axes = _grid(len(subjects), ncols, cell=(3.2, 2.8))
        for ax, s in zip(axes, subjects):
            g = emp[emp["subject_id"] == s].sort_values("bin")
            ax.fill_between(g["bin"] + 1, g["ci_lo"], g["ci_hi"], color=EMP_COLOR, alpha=0.12, lw=0)
            ax.plot(g["bin"] + 1, g["cbf"], color=EMP_COLOR, marker="D", ms=4, lw=2, label="Mouse")
            for m in MODELS:
                p = pred[(pred["subject_id"] == s) & (pred["model"] == m)].sort_values("bin")
                ax.plot(p["bin"] + 1, p["cbf"], "--", marker="o", ms=3, color=MODEL_COLORS[m], lw=1.4, label=m)
            ax.axhline(0, color="gray", ls=":", lw=0.8)
            w = winners.get(s)
            ax.set_title(s if w is None else f"{s} (best {w[0]}{'' if w[1] == 'decisive' else '?'})", fontsize=9)
            ax.set_xticks(range(1, int(g["bin"].max()) + 2))
        axes[0].legend(frameon=False, fontsize=7)
        for ax in axes[::ncols]:
            ax.set_ylabel("ΔP(Right)")
        fig.supxlabel("RT quantile bin (1 = fastest)  |  '?' = BIC winner not decisive")
    return _finish(fig, save_path)


def plot_cbf_slope_agreement(
    fit_table: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "CBF slope: model prediction vs mouse",
) -> plt.Figure:
    """Predicted against empirical CBF slope (change in ΔP(Right) per RT bin), one panel per model."""
    lim = float(np.nanmax(np.abs(fit_table[["slope_pred", "slope_emp"]].to_numpy()))) * 1.15
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=True, sharey=True)
        for ax, m in zip(axes, MODELS):
            d = fit_table[fit_table["model"] == m]
            ax.plot([-lim, lim], [-lim, lim], color="gray", ls=":", lw=1)
            ax.axhline(0, color="#cccccc", lw=0.8)
            ax.axvline(0, color="#cccccc", lw=0.8)
            ax.scatter(d["slope_emp"], d["slope_pred"], s=45, color=MODEL_COLORS[m], zorder=3)
            for _, r in d.iterrows():
                ax.annotate(r["subject_id"], (r["slope_emp"], r["slope_pred"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set(title=MODEL_LABELS[m], xlabel="Empirical CBF slope per bin", xlim=(-lim, lim), ylim=(-lim, lim))
        axes[0].set_ylabel("Model-predicted CBF slope per bin")
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)
