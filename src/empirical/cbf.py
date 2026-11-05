"""
Conditional Bias Function (CBF; White & Poldrack, 2014) for the inferred block prior.
Track B (Joshua Koilpillai).

    CBF(bin) = P(Right | prior Right, RT in bin) - P(Right | prior Left, RT in bin)

RT bins are within-mouse quantiles (quintiles by default). The CBF shape is the diagnostic
for the route by which the prior enters the DDM (plan §4, Week 2 / Week 3):

    M1 starting point  -> falling CBF   (bias strongest for fast RTs)
    M2 drift bias      -> flat CBF
    M3 dynamic bias    -> rising CBF    (bias strongest for slow RTs)

Contrast stratification
-----------------------
In the IBL task the stimulus side is correlated with the block (80/20), so the raw difference
P(R | R-block) - P(R | L-block) is inflated by the stimulus statistics themselves. This
implementation therefore computes the difference *within each signed contrast* and averages
over contrasts with weights n_R * n_L / (n_R + n_L) (inverse-variance for equal proportions).
By default only ambiguous stimuli (|contrast| <= 0.125) are used, where the prior matters
most; `max_abs_contrast=0` gives the purest estimate (0% contrast only).

The functions accept any §3.1 DataFrame, so the same code scores empirical mice and Track A's
simulated M1 / M2 / M3 data. Passing the empirical `bin_edges` to the simulated data puts both
on identical RT bins for the Week 3 overlay.
"""

from typing import Dict, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ._common import biased_trials, require_columns, sem

DEFAULT_MAX_ABS_CONTRAST = 0.125


def _select_frame(df: pd.DataFrame, max_abs_contrast: float, rt_col: str) -> pd.DataFrame:
    d = biased_trials(df)
    d = d[np.abs(d["contrast"]) <= max_abs_contrast + 1e-12]
    return d[np.isfinite(d[rt_col])]


def _select(df: pd.DataFrame, max_abs_contrast: float, rt_col: str):
    d = _select_frame(df, max_abs_contrast, rt_col)
    return (
        d["contrast"].to_numpy(dtype=float),
        (d["prior"].to_numpy() > 0),
        (d["choice"].to_numpy() == 1),
        d[rt_col].to_numpy(dtype=float),
    )


def _bin_edges(rt: np.ndarray, n_bins: int) -> np.ndarray:
    return np.quantile(rt, np.linspace(0.0, 1.0, n_bins + 1))


def _assign_bins(rt: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index 0..n_bins-1 using the interior edges (values beyond the range are clipped in)."""
    return np.searchsorted(edges[1:-1], rt, side="right")


def _cbf_core(contrast, right_prior, right_choice, rt, n_bins, edges=None, min_per_cell=1):
    """
    Vectorised stratified CBF. Returns (cbf, se, edges, counts) where counts has
    columns [n_trials, n_right_prior, n_left_prior] per bin.
    """
    if edges is None:
        edges = _bin_edges(rt, n_bins)
    bins = _assign_bins(rt, edges)
    strata, sidx = np.unique(contrast, return_inverse=True)
    ns = len(strata)
    # index = ((bin * ns + stratum) * 2 + is_right_prior)
    flat = (bins * ns + sidx) * 2 + right_prior.astype(int)
    size = n_bins * ns * 2
    n = np.bincount(flat, minlength=size).reshape(n_bins, ns, 2).astype(float)
    k = np.bincount(flat, weights=right_choice.astype(float), minlength=size).reshape(n_bins, ns, 2)
    nL, nR = n[..., 0], n[..., 1]
    valid = (nL >= min_per_cell) & (nR >= min_per_cell)
    with np.errstate(invalid="ignore", divide="ignore"):
        pL, pR = k[..., 0] / nL, k[..., 1] / nR
        delta = pR - pL
        w = np.where(valid, nR * nL / (nR + nL), 0.0)
        # Agresti-adjusted proportions for the variance only
        aL, aR = (k[..., 0] + 0.5) / (nL + 1.0), (k[..., 1] + 0.5) / (nR + 1.0)
        var = aR * (1 - aR) / nR + aL * (1 - aL) / nL
        wsum = w.sum(axis=1)
        cbf = np.where(wsum > 0, np.nansum(np.where(valid, w * delta, 0.0), axis=1) / wsum, np.nan)
        se = np.where(wsum > 0, np.sqrt(np.nansum(np.where(valid, w ** 2 * var, 0.0), axis=1)) / wsum, np.nan)
    counts = np.stack([n.sum(axis=(1, 2)), n[..., 1].sum(axis=1), n[..., 0].sum(axis=1)], axis=1)
    return cbf, se, edges, counts


def compute_cbf(
    df: pd.DataFrame,
    n_bins: int = 5,
    max_abs_contrast: float = DEFAULT_MAX_ABS_CONTRAST,
    bin_edges: Optional[np.ndarray] = None,
    rt_col: str = "rt",
    min_per_cell: int = 1,
) -> pd.DataFrame:
    """
    Contrast-stratified Conditional Bias Function over RT quantile bins for ONE dataset.

    Parameters
    ----------
    df : pd.DataFrame
        §3.1 trial table of a single mouse / simulation (pool mice yourself if desired).
    n_bins : int
        Number of RT quantile bins (5 = quintiles).
    max_abs_contrast : float
        Only trials with |contrast| <= this are used (0 -> 0% contrast only, 1 -> all).
    bin_edges : np.ndarray, optional
        Fixed RT bin edges of length n_bins + 1 (e.g. an empirical mouse's edges applied to
        simulated data). Default: quantiles of this dataset's selected trials.
    rt_col : str
        RT column (default 'rt'; 'rt_response' is available in the extended cohort file).
    min_per_cell : int
        Minimum trials per prior in a (bin, contrast) cell for the cell to contribute.

    Returns
    -------
    pd.DataFrame
        One row per bin: bin, rt_lo, rt_hi, rt_median, n_trials, n_right_prior,
        n_left_prior, cbf, se, ci_lo, ci_hi.
    """
    require_columns(df, ("contrast", "prior", "choice", rt_col))
    contrast, rp, rc, rt = _select(df, max_abs_contrast, rt_col)
    if len(rt) < n_bins * 2:
        raise ValueError(f"Too few usable trials ({len(rt)}) for {n_bins} RT bins")
    edges = np.asarray(bin_edges, dtype=float) if bin_edges is not None else None
    if edges is not None and len(edges) != n_bins + 1:
        raise ValueError(f"bin_edges must have length n_bins + 1 = {n_bins + 1}")
    cbf, se, edges, counts = _cbf_core(contrast, rp, rc, rt, n_bins, edges, min_per_cell)
    bins = _assign_bins(rt, edges)
    return pd.DataFrame({
        "bin": np.arange(n_bins),
        "rt_lo": edges[:-1],
        "rt_hi": edges[1:],
        "rt_median": [float(np.median(rt[bins == b])) if np.any(bins == b) else np.nan for b in range(n_bins)],
        "n_trials": counts[:, 0].astype(int),
        "n_right_prior": counts[:, 1].astype(int),
        "n_left_prior": counts[:, 2].astype(int),
        "cbf": cbf,
        "se": se,
        "ci_lo": cbf - 1.96 * se,
        "ci_hi": cbf + 1.96 * se,
    })


def cbf_curve(df: pd.DataFrame, n_bins: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    (RT bin medians, CBF values) tuple — drop-in replacement for Track A's `compute_cbf_curve`
    so that `plot_cbf({...: cbf_curve(df)})` works with the stratified estimator.
    """
    tab = compute_cbf(df, n_bins=n_bins, **kwargs)
    return tab["rt_median"].to_numpy(), tab["cbf"].to_numpy()


def cbf_slope(values: Sequence[float]) -> float:
    """OLS slope of CBF against bin index (change in ΔP(Right) per RT quantile bin)."""
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


# Smallest CBF change per RT bin treated as a real trend. Calibrated against Track A's simulator:
# a pure drift-bias model (M2, default demo parameters) yields a small slope of about -0.027 per bin
# (20k trials), whereas M1 gives about -0.25 and M3 about +0.09, so the floor must exceed M2's.
DEFAULT_MIN_ABS_SLOPE = 0.03


def classify_shape(
    slope: float,
    slope_lo: float,
    slope_hi: float,
    min_abs_slope: float = DEFAULT_MIN_ABS_SLOPE,
) -> str:
    """
    Label a CBF by its slope across RT bins, requiring both statistical and practical support.

    'falling' (M1-like): bootstrap CI entirely below 0 and slope <= -min_abs_slope.
    'rising'  (M3-like): bootstrap CI entirely above 0 and slope >= +min_abs_slope.
    otherwise 'flat' (M2-like): no reliable trend of at least `min_abs_slope` per bin
    (absence of evidence for a trend, not proof of flatness).

    The effect-size floor stops very large samples from labelling a nearly flat CBF
    (e.g. the small decline a pure drift-bias DDM produces in its slowest bin) as falling.
    """
    if not (np.isfinite(slope) and np.isfinite(slope_lo) and np.isfinite(slope_hi)):
        return "undetermined"
    if slope_hi < 0 and slope <= -min_abs_slope:
        return "falling"
    if slope_lo > 0 and slope >= min_abs_slope:
        return "rising"
    return "flat"


def bootstrap_cbf(
    df: pd.DataFrame,
    n_boot: int = 500,
    n_bins: int = 5,
    max_abs_contrast: float = DEFAULT_MAX_ABS_CONTRAST,
    rt_col: str = "rt",
    seed: int = 0,
    min_abs_slope: float = DEFAULT_MIN_ABS_SLOPE,
    cluster_col: Optional[str] = None,
) -> Dict[str, object]:
    """
    Bootstrap of the CBF (bin edges are re-estimated in every resample).

    By default trials are resampled independently. With `cluster_col` (e.g. 'session_eid')
    whole clusters are resampled instead, which respects the within-session dependence of
    trials (block structure, drifting engagement) and gives wider, more honest intervals.

    Returns
    -------
    dict
        cbf (observed), cbf_lo / cbf_hi (95% percentile bands), slope, slope_lo, slope_hi,
        shape (see `classify_shape`), n_trials, boot_cbf (n_boot x n_bins), boot_slope.
    """
    require_columns(df, ("contrast", "prior", "choice", rt_col) + ((cluster_col,) if cluster_col else ()))
    d = _select_frame(df, max_abs_contrast, rt_col)
    contrast = d["contrast"].to_numpy(dtype=float)
    rp, rc, rt = d["prior"].to_numpy() > 0, d["choice"].to_numpy() == 1, d[rt_col].to_numpy(dtype=float)
    n = len(rt)
    if n < n_bins * 2:
        raise ValueError(f"Too few usable trials ({n}) for {n_bins} RT bins")
    obs, *_ = _cbf_core(contrast, rp, rc, rt, n_bins)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_bins))
    if cluster_col is not None:
        codes, _ = pd.factorize(d[cluster_col])
        members = [np.flatnonzero(codes == j) for j in range(int(codes.max()) + 1)]
    for b in range(n_boot):
        if cluster_col is None:
            i = rng.integers(0, n, n)
        else:
            i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        boot[b] = _cbf_core(contrast[i], rp[i], rc[i], rt[i], n_bins)[0]
    slopes = np.array([cbf_slope(r) for r in boot])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins when a prior block is absent
        s_lo, s_hi = np.nanpercentile(slopes, [2.5, 97.5]) if np.isfinite(slopes).any() else (np.nan, np.nan)
        cbf_lo, cbf_hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    slope_obs = cbf_slope(obs)
    return {
        "cbf": obs,
        "cbf_lo": cbf_lo,
        "cbf_hi": cbf_hi,
        "slope": slope_obs,
        "slope_lo": float(s_lo),
        "slope_hi": float(s_hi),
        "shape": classify_shape(slope_obs, s_lo, s_hi, min_abs_slope),
        "n_trials": int(n),
        "boot_cbf": boot,
        "boot_slope": slopes,
    }


def cbf_by_subject(
    df: pd.DataFrame,
    n_bins: int = 5,
    max_abs_contrast: float = DEFAULT_MAX_ABS_CONTRAST,
    n_boot: int = 500,
    rt_col: str = "rt",
    seed: int = 0,
    min_abs_slope: float = DEFAULT_MIN_ABS_SLOPE,
    cluster_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CBF for every mouse (`subject_id`) with bootstrap intervals and shape classification.
    `cluster_col` switches the bootstrap to whole-cluster (e.g. session) resampling.

    Returns
    -------
    curves : pd.DataFrame
        Long table: subject_id, bin, rt_median, cbf, se, cbf_lo, cbf_hi, n_trials.
    slopes : pd.DataFrame
        One row per mouse: subject_id, n_trials, slope, slope_lo, slope_hi, shape.
    """
    require_columns(df, ("subject_id", "contrast", "prior", "choice", rt_col))
    curves, slopes = [], []
    for i, (subj, g) in enumerate(df.groupby("subject_id", observed=True)):
        tab = compute_cbf(g, n_bins, max_abs_contrast, rt_col=rt_col)
        bs = bootstrap_cbf(g, n_boot, n_bins, max_abs_contrast, rt_col, seed=seed + i,
                           min_abs_slope=min_abs_slope, cluster_col=cluster_col)
        tab = tab.assign(subject_id=subj, cbf_lo=bs["cbf_lo"], cbf_hi=bs["cbf_hi"])
        curves.append(tab[["subject_id", "bin", "rt_median", "n_trials", "cbf", "se", "cbf_lo", "cbf_hi"]])
        slopes.append({"subject_id": subj, "n_trials": bs["n_trials"], "slope": bs["slope"],
                       "slope_lo": bs["slope_lo"], "slope_hi": bs["slope_hi"], "shape": bs["shape"]})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(slopes)


def cohort_cbf(curves: pd.DataFrame) -> pd.DataFrame:
    """Cohort CBF per bin: mean and SEM across mice (equal weights) from `cbf_by_subject` curves."""
    out = (
        curves.groupby("bin")
        .agg(rt_median=("rt_median", "mean"), cbf=("cbf", "mean"),
             sem=("cbf", lambda s: sem(s.to_numpy())), n_subjects=("cbf", "count"))
        .reset_index()
    )
    return out


def cohort_slope_test(slopes: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """
    Across-mouse test of the CBF slope (H0: mean slope = 0) with counts of per-mouse shapes.
    `exclude` drops the named mice first (leave-one-out robustness checks).
    """
    if exclude:
        slopes = slopes[~slopes["subject_id"].isin(list(exclude))]
    s = slopes["slope"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    out: Dict[str, object] = {
        "n_mice": int(len(s)),
        "mean_slope": float(np.mean(s)) if len(s) else np.nan,
        "sem_slope": sem(s),
        "shape_counts": slopes["shape"].value_counts().to_dict(),
    }
    if len(s) >= 3:
        out["t_stat"], out["t_p"] = (float(v) for v in stats.ttest_1samp(s, 0.0))
        try:
            out["wilcoxon_stat"], out["wilcoxon_p"] = (float(v) for v in stats.wilcoxon(s))
        except ValueError:  # all-zero differences
            out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    return out
