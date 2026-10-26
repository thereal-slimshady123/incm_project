"""
Publication-Grade Visualizations for Computational Modeling Recovery (Wilson & Collins, 2019).

Generates:
1. True vs. Recovered Parameter Scatterplots with Pearson r, RMSE, and slope.
2. Parameter Error Correlation Matrix (Corr(Delta theta_i, Delta theta_j)).
3. Model Recovery Confusion and Inversion Matrices (BIC-based).
4. Diagnostic Conditional Bias Functions (CBF; White & Poldrack, 2014).
5. Psychometric (accuracy) & Chronometric (RT) curves.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


# Set clean, publication-grade matplotlib style
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


def plot_parameter_recovery(
    recovery_df: pd.DataFrame,
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Wilson & Collins Parameter Recovery (Rule 4)",
) -> plt.Figure:
    """
    Plot ground-truth vs. recovered parameters with identity line,
    linear regression slope, Pearson r, and RMSE.
    """
    if param_names is None:
        param_names = [col.replace("true_", "") for col in recovery_df.columns if col.startswith("true_")]

    n_params = len(param_names)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for idx, p in enumerate(param_names):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        true_col = f"true_{p}"
        fit_col = f"fit_{p}"

        if true_col not in recovery_df.columns or fit_col not in recovery_df.columns:
            ax.set_visible(False)
            continue

        true_vals = recovery_df[true_col].values
        fit_vals = recovery_df[fit_col].values

        # Remove NaNs if any
        valid = np.isfinite(true_vals) & np.isfinite(fit_vals)
        true_vals = true_vals[valid]
        fit_vals = fit_vals[valid]

        if len(true_vals) < 2:
            ax.set_title(p)
            continue

        # Statistics
        r_val, _ = pearsonr(true_vals, fit_vals)
        rmse = np.sqrt(np.mean((true_vals - fit_vals) ** 2))
        slope, intercept, _, _, _ = linregress(true_vals, fit_vals)

        # Plot scatter
        ax.scatter(true_vals, fit_vals, alpha=0.75, color="#1f77b4", edgecolor="#0e436b", s=36, zorder=3)

        # Identity line
        min_v = min(np.min(true_vals), np.min(fit_vals))
        max_v = max(np.max(true_vals), np.max(fit_vals))
        margin = 0.05 * (max_v - min_v if max_v > min_v else 1.0)
        line_range = np.linspace(min_v - margin, max_v + margin, 50)
        ax.plot(line_range, line_range, "k--", alpha=0.6, label="Identity (y=x)", zorder=2)

        # Fitted regression line
        reg_line = slope * line_range + intercept
        ax.plot(line_range, reg_line, color="#d62728", lw=1.8, label=f"Fit (slope={slope:.2f})", zorder=2)

        ax.set_xlim(min_v - margin, max_v + margin)
        ax.set_ylim(min_v - margin, max_v + margin)
        ax.set_xlabel(f"True {p}", fontsize=11)
        ax.set_ylabel(f"Recovered {p}", fontsize=11)
        ax.set_title(f"Parameter: {p}\n$r={r_val:.3f}$, $\\mathrm{{RMSE}}={rmse:.3f}$", fontsize=11)
        ax.grid(True)
        ax.legend(frameon=True, fontsize=8, loc="upper left")

    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_error_correlation_matrix(
    recovery_df: pd.DataFrame,
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Parameter Estimation Error Correlation Matrix: Corr(Δθ_i, Δθ_j)",
) -> plt.Figure:
    """
    Computes and plots the error correlation matrix Corr(Delta theta_i, Delta theta_j).
    Crucial for Wilson & Collins Rule 4 to rule out trade-offs / cross-talk (e.g. z vs. v_b).
    """
    if param_names is None:
        param_names = [col.replace("true_", "") for col in recovery_df.columns if col.startswith("true_")]

    # Calculate parameter errors: Delta theta = theta_fit - theta_true
    error_dict = {}
    for p in param_names:
        true_col = f"true_{p}"
        fit_col = f"fit_{p}"
        if true_col in recovery_df.columns and fit_col in recovery_df.columns:
            error_dict[f"Δ{p}"] = recovery_df[fit_col] - recovery_df[true_col]

    error_df = pd.DataFrame(error_dict).dropna()
    corr_matrix = error_df.corr().values
    labels = list(error_df.columns)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=11)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate values inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_model_recovery_confusion(
    confusion_counts: np.ndarray,
    model_labels: List[str] = ("M1 (Start)", "M2 (Drift)", "M3 (Dynamic)"),
    save_path: Optional[str] = None,
    title_prefix: str = "Wilson & Collins Model Recovery (Rule 5)",
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates side-by-side Confusion Matrix P(Fit | True) and Inversion Matrix P(True | Fit).
    Demonstrates absence of model mimicry.
    """
    # 1. Confusion Matrix: Normalized by Row (True Model)
    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    conf_matrix = np.divide(confusion_counts, np.maximum(row_sums, 1))

    # 2. Inversion Matrix: Normalized by Column (Fit Model)
    col_sums = confusion_counts.sum(axis=0, keepdims=True)
    inv_matrix = np.divide(confusion_counts, np.maximum(col_sums, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # Plot Confusion Matrix
    im1 = ax1.imshow(conf_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title(f"{title_prefix}\nConfusion Matrix: P(Fit | True)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Fitted Model (Min BIC)", fontsize=11)
    ax1.set_ylabel("True Generative Model", fontsize=11)
    ax1.set_xticks(range(len(model_labels)))
    ax1.set_yticks(range(len(model_labels)))
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(model_labels)):
        for j in range(len(model_labels)):
            val = conf_matrix[i, j]
            count = confusion_counts[i, j]
            color = "white" if val > 0.5 else "black"
            ax1.text(j, i, f"{val:.1%}\n(N={count})", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Plot Inversion Matrix
    im2 = ax2.imshow(inv_matrix, cmap="Greens", vmin=0.0, vmax=1.0)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title(f"{title_prefix}\nInversion Matrix: P(True | Fit)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Fitted Model (Min BIC)", fontsize=11)
    ax2.set_ylabel("True Generative Model", fontsize=11)
    ax2.set_xticks(range(len(model_labels)))
    ax2.set_yticks(range(len(model_labels)))
    ax2.set_xticklabels(model_labels, fontsize=10)
    ax2.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(model_labels)):
        for j in range(len(model_labels)):
            val = inv_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def compute_cbf_curve(trials_df: pd.DataFrame, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Conditional Bias Function (CBF; White & Poldrack, 2014):
        Delta P(Right) = P(Right | Prior Right) - P(Right | Prior Left)
    evaluated across RT quantiles.
    """
    # Use zero or ambiguous contrasts for purest prior bias measurement
    sub = trials_df[np.abs(trials_df["contrast"]) <= 0.125].copy()
    if len(sub) < 30:
        sub = trials_df.copy()

    # Determine RT bins across all qualifying trials
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sub["rt"].values, quantiles)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cbf_values = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = sub[(sub["rt"] >= low) & (sub["rt"] <= high)]

        right_prior = in_bin[in_bin["prior"] > 0]
        left_prior = in_bin[in_bin["prior"] < 0]

        p_right_given_rp = np.mean(right_prior["choice"] == 1) if len(right_prior) > 0 else np.nan
        p_right_given_lp = np.mean(left_prior["choice"] == 1) if len(left_prior) > 0 else np.nan

        delta_p = p_right_given_rp - p_right_given_lp
        cbf_values.append(delta_p)

    return bin_centers, np.array(cbf_values)


def plot_cbf(
    cbf_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "Diagnostic Conditional Bias Functions (CBF; White & Poldrack, 2014)",
) -> plt.Figure:
    """
    Overlays CBF curves:
    - M1 (Starting Point): Falling bias with RT
    - M2 (Drift Bias): Flat bias across RT
    - M3 (Dynamic Bias): Rising bias with RT
    - Empirical (if provided): mouse data
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    colors = {
        "M1 (Starting Point)": "#1f77b4",
        "M2 (Drift Bias)": "#2ca02c",
        "M3 (Dynamic Bias)": "#d62728",
        "Empirical Mouse Data": "#111111",
    }
    markers = {
        "M1 (Starting Point)": "o",
        "M2 (Drift Bias)": "s",
        "M3 (Dynamic Bias)": "^",
        "Empirical Mouse Data": "D",
    }

    for label, (rt_centers, cbf_vals) in cbf_dict.items():
        c = colors.get(label, "#555555")
        m = markers.get(label, "o")
        lw = 2.4 if "Empirical" in label else 1.8
        ls = "-" if "Empirical" in label else "--"

        ax.plot(rt_centers, cbf_vals, color=c, marker=m, lw=lw, linestyle=ls, label=label, markersize=7)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Reaction Time (s)", fontsize=11)
    ax.set_ylabel("Prior Bias ΔP(Right | Prior)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=True, fontsize=10, loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_psychometric_chronometric(
    df: pd.DataFrame,
    sim_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots psychometric sigmoid (P(Right) vs signed contrast) and chronometric curve (mean RT vs contrast).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for prior_val, color, lbl in [(1.0, "#2b83ba", "Right Prior (80/20)"), (-1.0, "#d7191c", "Left Prior (20/80)")]:
        sub = df[df["prior"] == prior_val]
        grouped = sub.groupby("contrast", observed=False)
        contrasts = grouped["choice"].mean().index.values
        p_right = grouped["choice"].apply(lambda s: np.mean(s == 1)).values
        mean_rt = grouped["rt"].mean().values

        ax1.plot(contrasts, p_right, "o", color=color, label=f"Data: {lbl}", markersize=6)
        ax2.plot(contrasts, mean_rt, "o", color=color, label=f"Data: {lbl}", markersize=6)

        if sim_df is not None:
            sim_sub = sim_df[sim_df["prior"] == prior_val]
            sim_grouped = sim_sub.groupby("contrast", observed=False)
            sim_c = sim_grouped["choice"].mean().index.values
            sim_p_right = sim_grouped["choice"].apply(lambda s: np.mean(s == 1)).values
            sim_mean_rt = sim_grouped["rt"].mean().values

            ax1.plot(sim_c, sim_p_right, "-", color=color, lw=2.0, alpha=0.8, label=f"Model: {lbl}")
            ax2.plot(sim_c, sim_mean_rt, "-", color=color, lw=2.0, alpha=0.8, label=f"Model: {lbl}")

    ax1.set_xlabel("Signed Contrast", fontsize=11)
    ax1.set_ylabel("P(Choose Right)", fontsize=11)
    ax1.set_title("Psychometric Function", fontsize=11, fontweight="bold")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Signed Contrast", fontsize=11)
    ax2.set_ylabel("Mean Reaction Time (s)", fontsize=11)
    ax2.set_title("Chronometric Function", fontsize=11, fontweight="bold")
    ax2.grid(True)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
