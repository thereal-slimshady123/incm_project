"""
Individual-mouse heterogeneity profiling (Track B, Joshua Koilpillai).

Collapses each mouse's behaviour into one row of interpretable metrics so that between-mouse
differences (and later, differences in the winning prior mechanism) can be related to simple
behavioural phenotypes:

    sensitivity        sigma (psychometric slope scale), lapses gamma / lam, easy / hard accuracy
    prior use          prior_shift (contrast units), prior_effect_c0 (P(R) difference at 0% contrast),
                       gamma_block_diff / lam_block_diff (the prior acting on the lapse asymptotes)
    history            repeat_rate, history_effect_c0 (P(R | prev R) - P(R | prev L) at 0% contrast,
                       computed WITHIN each block so it is not confounded with the block prior)
    speed              RT median / IQR / q90, fraction of slow trials, chronometric slope
    CBF                slope across RT quintiles, bootstrap CI and shape label
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._common import biased_trials, require_columns
from .cbf import cbf_by_subject
from .psychometrics import fit_psychometric_by, prior_shift

EASY_CONTRAST = 0.25
HARD_CONTRAST = 0.125
SLOW_RT = 1.0  # seconds

PROFILE_METRICS: List[str] = [
    "sigma", "gamma", "lam", "easy_accuracy", "hard_accuracy",
    "prior_shift", "prior_effect_c0", "gamma_block_diff", "lam_block_diff",
    "repeat_rate", "history_effect_c0",
    "rt_median", "rt_iqr", "rt_q90", "frac_slow", "chrono_slope", "cbf_slope",
]


def _chrono_slope(g: pd.DataFrame) -> float:
    """Slope of mean RT against |contrast| (seconds per unit contrast; negative = faster for easy)."""
    m = g.groupby(np.abs(g["contrast"]))["rt"].mean()
    if len(m) < 3:
        return np.nan
    return float(np.polyfit(m.index.to_numpy(dtype=float), m.to_numpy(), 1)[0])


def _history_effect_c0(hist: pd.DataFrame) -> float:
    """
    P(R | prev R) - P(R | prev L) at 0% contrast, estimated within each block prior and averaged
    over the blocks, so that the block (which correlates with consecutive choices) cannot
    masquerade as a history effect. NaN when no block has both previous-choice values.
    """
    diffs = []
    for _, blk in hist[hist["contrast"] == 0].groupby("prior"):
        pr = (blk.loc[blk["prev_choice"] > 0, "choice"] == 1).mean()
        pl = (blk.loc[blk["prev_choice"] < 0, "choice"] == 1).mean()
        if np.isfinite(pr) and np.isfinite(pl):
            diffs.append(pr - pl)
    return float(np.mean(diffs)) if diffs else np.nan


def mouse_profile(
    df: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 0,
    shifts: Optional[pd.DataFrame] = None,
    slopes: Optional[pd.DataFrame] = None,
    cluster_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    One row of behavioural metrics per mouse (`subject_id`) from a §3.1 trial table.

    `prev_choice` is used for the history metrics if present. Bootstrap intervals are included
    for the prior shift and the CBF slope (session-level resampling if `cluster_col` is given).
    `gamma` / `lam` are the mean over the two blocks; `gamma_block_diff` / `lam_block_diff` are
    right-block minus left-block, i.e. how much the prior moves each lapse asymptote. Pass precomputed `shifts` (from `prior_shift`) and
    `slopes` (second output of `cbf_by_subject`) to avoid repeating the bootstraps.
    """
    require_columns(df)
    d = biased_trials(df)
    fits = fit_psychometric_by(d, by=("subject_id", "prior"))
    if shifts is None:
        shifts = prior_shift(d, n_boot=n_boot, seed=seed, cluster_col=cluster_col)
    if slopes is None:
        _, slopes = cbf_by_subject(d, n_boot=n_boot, seed=seed, cluster_col=cluster_col)
    shifts = shifts.set_index("subject_id")
    slopes = slopes.set_index("subject_id")

    rows: List[Dict[str, object]] = []
    for subj, g in d.groupby("subject_id", observed=True):
        f = fits[fits["subject_id"] == subj]
        nz = g[g["contrast"] != 0]
        easy = nz[np.abs(nz["contrast"]) >= EASY_CONTRAST]
        hard = nz[np.abs(nz["contrast"]) <= HARD_CONTRAST]
        acc = lambda x: float((np.sign(x["contrast"]) == x["choice"]).mean()) if len(x) else np.nan  # noqa: E731
        q = g["rt"].quantile([0.25, 0.5, 0.75, 0.9])
        row: Dict[str, object] = {
            "subject_id": subj,
            "n_trials": int(len(g)),
            "p_right": float((g["choice"] == 1).mean()),
            "sigma": float(f["sigma"].mean()),
            "gamma": float(f["gamma"].mean()),
            "lam": float(f["lam"].mean()),
            "gamma_right_block": float(shifts.loc[subj, "gamma_right_block"]),
            "gamma_left_block": float(shifts.loc[subj, "gamma_left_block"]),
            "lam_right_block": float(shifts.loc[subj, "lam_right_block"]),
            "lam_left_block": float(shifts.loc[subj, "lam_left_block"]),
            "gamma_block_diff": float(shifts.loc[subj, "gamma_right_block"] - shifts.loc[subj, "gamma_left_block"]),
            "lam_block_diff": float(shifts.loc[subj, "lam_right_block"] - shifts.loc[subj, "lam_left_block"]),
            "easy_accuracy": acc(easy),
            "hard_accuracy": acc(hard),
            "prior_shift": float(shifts.loc[subj, "shift"]),
            "prior_shift_lo": float(shifts.loc[subj, "shift_lo"]) if "shift_lo" in shifts else np.nan,
            "prior_shift_hi": float(shifts.loc[subj, "shift_hi"]) if "shift_hi" in shifts else np.nan,
            "prior_effect_c0": float(shifts.loc[subj, "prior_effect_c0"]),
            "rt_median": float(q[0.5]),
            "rt_iqr": float(q[0.75] - q[0.25]),
            "rt_q90": float(q[0.9]),
            "frac_slow": float((g["rt"] > SLOW_RT).mean()),
            "chrono_slope": _chrono_slope(g),
            "cbf_slope": float(slopes.loc[subj, "slope"]),
            "cbf_slope_lo": float(slopes.loc[subj, "slope_lo"]),
            "cbf_slope_hi": float(slopes.loc[subj, "slope_hi"]),
            "cbf_shape": slopes.loc[subj, "shape"],
        }
        if "prev_choice" in g.columns:
            hist = g[g["prev_choice"] != 0]
            row["repeat_rate"] = float((hist["choice"] == hist["prev_choice"]).mean()) if len(hist) else np.nan
            row["history_effect_c0"] = _history_effect_c0(hist)
        else:
            row["repeat_rate"] = row["history_effect_c0"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def heterogeneity_summary(profile: pd.DataFrame, metrics: List[str] = PROFILE_METRICS) -> pd.DataFrame:
    """
    Between-mouse spread of each metric: mean, SD, coefficient of variation, range, and the
    mice whose robust z-score (median / MAD) exceeds 2 in absolute value.
    """
    rows = []
    for m in metrics:
        v = profile[m].astype(float)
        mad = np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.4826
        z = (v - np.nanmedian(v)) / mad if mad > 0 else pd.Series(np.zeros(len(v)), index=v.index)
        mean = float(np.nanmean(v))
        rows.append({
            "metric": m, "mean": mean, "sd": float(np.nanstd(v, ddof=1)),
            "cv": float(np.nanstd(v, ddof=1) / abs(mean)) if mean != 0 else np.nan,
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
            "outliers": ", ".join(profile.loc[z.abs() > 2, "subject_id"].astype(str)),
        })
    return pd.DataFrame(rows)


def profile_correlations(profile: pd.DataFrame, metrics: List[str] = PROFILE_METRICS) -> pd.DataFrame:
    """Spearman correlation matrix between metrics across mice (descriptive; n is small)."""
    return profile[metrics].astype(float).corr(method="spearman")
