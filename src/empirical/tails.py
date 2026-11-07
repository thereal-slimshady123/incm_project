"""
Model-free RT-tail diagnostics for the urgency question (C2).

The hazard h(t) is the chance that a trial still running at time t ends in the next instant. In a diffusion model with a FIXED
bound it rises with t and levels off at a constant (the RT tail is exponential); a COLLAPSING bound (urgency) squeezes the late
tail, so the hazard keeps RISING well beyond that level and late responses are driven more by noise and the prior than by the
stimulus (accuracy falls toward chance). Trial-to-trial drift variability or disengaged trials do the opposite: a mixture of
slow processes lowers the late hazard. A late/early hazard ratio above 1 therefore does NOT by itself indicate urgency (a fixed
bound gives about 1.2 for the mice's typical parameters); read it against the ratio the fitted fixed-bound model predicts
(`src.optimization.c2.tail_errors`), and note that very steep collapse leaves few late trials and a noisy ratio.

Every function takes any §3.1 trial table (IBL data or the output of the simulator), so the same code
scores the mice and the model-predicted data. RTs are the contract `rt` (seconds after stimulus onset).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import require_columns

DEFAULT_SURVIVAL_GRID = tuple(np.round(np.arange(0.5, 4.51, 0.25), 2))
DEFAULT_HAZARD_EDGES = tuple(np.round(np.arange(0.5, 4.51, 0.5), 2))
DEFAULT_ACCURACY_EDGES = (0.1, 0.5, 1.0, 2.0, 3.5, 5.0)
EARLY_WINDOW = (1.0, 2.0)
LATE_WINDOW = (2.5, 4.0)


def _groups(df: pd.DataFrame, by: Sequence[str]):
    """Yield (key dict, frame); `by=()` treats the table as one group."""
    if not by:
        yield {}, df
        return
    for key, g in df.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(by, key)), g


def survival_table(df: pd.DataFrame, grid: Sequence[float] = DEFAULT_SURVIVAL_GRID,
                   by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """P(RT > t) of the trials in the table at every t of `grid`, per group. Columns: *by, t, survival, n."""
    require_columns(df, ("rt",))
    rows = []
    for key, g in _groups(df, by):
        rt = np.sort(g["rt"].to_numpy(dtype=float))
        for t in grid:
            rows.append({**key, "t": float(t), "survival": float(1.0 - np.searchsorted(rt, t, side="right") / len(rt)), "n": len(rt)})
    return pd.DataFrame(rows)


def _window_rate(events: int, at_risk: int, width: float) -> float:
    """
    Average hazard (1/s) of a window from its life-table counts: -ln(1 - events/at_risk) / width, the exact value for a
    piecewise-constant hazard (the naive events / at_risk / width understates the hazard of wide windows: 0.79 instead of 1.0
    for a true rate of 1/s and a 0.5 s window). If every trial at risk ends in the window the fraction is capped at
    1 - 1/(2 at_risk) so the estimate stays finite.
    """
    if at_risk <= 0:
        return float("nan")
    frac = min(events / at_risk, 1.0 - 0.5 / at_risk)
    return float(-np.log1p(-frac) / width)


def hazard_table(df: pd.DataFrame, edges: Sequence[float] = DEFAULT_HAZARD_EDGES,
                 by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """
    Average hazard per window [lo, hi) (see `_window_rate`): responses ending in the window among the trials still running at
    `lo`. Columns: *by, t_lo, t_hi, n_at_risk, n_events, hazard (units 1/s).
    """
    require_columns(df, ("rt",))
    edges = np.asarray(edges, dtype=float)
    rows = []
    for key, g in _groups(df, by):
        rt = g["rt"].to_numpy(dtype=float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            at_risk = int((rt >= lo).sum())
            events = int(((rt >= lo) & (rt < hi)).sum())
            rows.append({**key, "t_lo": float(lo), "t_hi": float(hi), "n_at_risk": at_risk, "n_events": events,
                         "hazard": _window_rate(events, at_risk, hi - lo)})
    return pd.DataFrame(rows)


def window_hazard(rt: np.ndarray, window: Tuple[float, float]) -> float:
    """Average hazard (1/s) over one window [lo, hi) of an RT vector (see `_window_rate`)."""
    lo, hi = window
    return _window_rate(int(((rt >= lo) & (rt < hi)).sum()), int((rt >= lo).sum()), hi - lo)


def late_hazard_ratio(df: pd.DataFrame, early: Tuple[float, float] = EARLY_WINDOW, late: Tuple[float, float] = LATE_WINDOW,
                      by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """
    hazard(late window = 2.5-4 s) / hazard(early window = 1-2 s) per group. A fixed bound already gives a ratio somewhat above 1
    (hazard rising towards its asymptote); a collapsing bound inflates it well beyond that (several-fold); a mixture of slow
    processes (drift variability, disengaged trials) pulls it down towards or below 1. Compare data with the fitted fixed-bound
    model, not with 1. Columns: *by, hazard_early, hazard_late, hazard_ratio, n_late_events.
    """
    require_columns(df, ("rt",))
    rows = []
    for key, g in _groups(df, by):
        rt = g["rt"].to_numpy(dtype=float)
        h_e, h_l = window_hazard(rt, early), window_hazard(rt, late)
        rows.append({**key, "hazard_early": h_e, "hazard_late": h_l, "hazard_ratio": h_l / h_e if h_e > 0 else np.nan,
                     "n_late_events": int(((rt >= late[0]) & (rt < late[1])).sum())})
    return pd.DataFrame(rows)


def tail_fraction(df: pd.DataFrame, cutoffs: Sequence[float] = (1.0, 2.0, 2.5, 3.5),
                  by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """Fraction of trials slower than each cutoff, in wide format (columns p_gt_<cutoff>)."""
    require_columns(df, ("rt",))
    rows = []
    for key, g in _groups(df, by):
        rt = g["rt"].to_numpy(dtype=float)
        rows.append({**key, "n": len(rt), **{f"p_gt_{c:g}": float((rt > c).mean()) for c in cutoffs}})
    return pd.DataFrame(rows)


def accuracy_by_rt_window(df: pd.DataFrame, edges: Sequence[float] = DEFAULT_ACCURACY_EDGES,
                          min_abs_contrast: float = 0.25, by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """
    Accuracy on clearly visible stimuli (|contrast| >= min_abs_contrast) in FIXED RT windows [lo, hi)
    (the last one closed), so that data and model windows are identical. A collapsing bound predicts that
    accuracy falls toward 0.5 in the late windows; a fixed bound keeps it up. Columns: *by, window, t_lo, t_hi, n, accuracy.
    """
    require_columns(df)
    d = df[np.abs(df["contrast"]) >= min_abs_contrast]
    edges = np.asarray(edges, dtype=float)
    rows = []
    for key, g in _groups(d, by):
        rt = g["rt"].to_numpy(dtype=float)
        correct = (np.sign(g["contrast"].to_numpy()) == g["choice"].to_numpy()).astype(float)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (rt >= lo) & ((rt < hi) if i < len(edges) - 2 else (rt <= hi))
            rows.append({**key, "window": i, "t_lo": float(lo), "t_hi": float(hi), "n": int(m.sum()),
                         "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def tail_summary(df: pd.DataFrame, by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """One row per group: tail fractions, early/late hazards, their ratio and early/late accuracy (|c| >= 0.25)."""
    keys = list(by)
    if not keys:
        raise ValueError("tail_summary needs at least one grouping column")
    out = tail_fraction(df, by=by).merge(late_hazard_ratio(df, by=by), on=keys)
    acc = accuracy_by_rt_window(df, edges=(0.1, 1.0, 2.0, 5.0), by=by)
    for window, name in ((0, "accuracy_lt1s"), (2, "accuracy_gt2s")):
        sub = acc[acc["window"] == window][keys + ["accuracy"]].rename(columns={"accuracy": name})
        out = out.merge(sub, on=keys, how="left")
    out["accuracy_drop"] = out["accuracy_lt1s"] - out["accuracy_gt2s"]
    return out


def survival_distance(data: pd.DataFrame, model: pd.DataFrame, t_min: float = 1.0) -> float:
    """
    Mean absolute difference between the survival curves of two (survival_table) outputs for ONE group,
    over t >= t_min. Both tables are on the same grid; the tail error used to rank fixed vs collapsing fits.
    """
    a = data[data["t"] >= t_min].sort_values("t")["survival"].to_numpy()
    b = model[model["t"] >= t_min].sort_values("t")["survival"].to_numpy()
    return float(np.mean(np.abs(a - b)))
