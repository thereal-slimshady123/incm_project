"""
Empirical psychometric and chronometric analysis (Track B, Joshua Koilpillai).

- Psychometric curves:  P(Right) vs signed contrast, with Wilson intervals and a 4-parameter
  lapse-corrected logistic fit  P(R | c) = gamma + (1 - gamma - lam) / (1 + exp(-(c - mu) / sigma))
  (mu = point of subjective equality, sigma = slope scale; gamma = P(Right) for strongly LEFT
  stimuli, i.e. the lapse / error rate on left stimuli; lam = P(Left) for strongly RIGHT stimuli).
- Prior shift:          horizontal shift of the psychometric curve between 80/20 and 20/80 blocks.
- Chronometric curves:  mean / median RT vs signed contrast (optionally split by outcome).
- Conditional accuracy: accuracy as a function of RT bin (speed-accuracy diagnostics).

Every function takes any §3.1 trial DataFrame (empirical, simulated or mock).
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from ._common import biased_trials, require_columns, sem, wilson_interval

_BOUNDS = [(-1.0, 1.0), (0.01, 1.5), (0.0, 0.5), (0.0, 0.5)]
_STARTS = [
    (0.0, 0.15, 0.05, 0.05),
    (0.0, 0.05, 0.02, 0.02),
    (0.1, 0.30, 0.10, 0.10),
    (-0.1, 0.10, 0.00, 0.00),
]


def psychometric_function(c, mu: float, sigma: float, gamma: float, lam: float) -> np.ndarray:
    """Lapse-corrected logistic psychometric function P(Right | signed contrast)."""
    c = np.asarray(c, dtype=float)
    return gamma + (1.0 - gamma - lam) * expit((c - mu) / sigma)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def psychometric_table(df: pd.DataFrame, by: Sequence[str] = ("subject_id", "prior")) -> pd.DataFrame:
    """
    P(Right) per signed contrast and grouping keys, with 95% Wilson intervals.

    Returns columns: *by, contrast, n, n_right, p_right, ci_lo, ci_hi.
    Trials with the neutral prior (0) are dropped when grouping by prior.
    """
    require_columns(df)
    d = biased_trials(df) if "prior" in by else df
    d = d.assign(_right=(d["choice"] == 1).astype(int))
    out = (
        d.groupby(list(by) + ["contrast"], observed=True)
        .agg(n=("_right", "size"), n_right=("_right", "sum"))
        .reset_index()
    )
    out["p_right"] = out["n_right"] / out["n"]
    out["ci_lo"], out["ci_hi"] = wilson_interval(out["n_right"], out["n"])
    return out


def chronometric_table(
    df: pd.DataFrame,
    by: Sequence[str] = ("subject_id", "prior"),
    by_outcome: bool = False,
) -> pd.DataFrame:
    """
    Mean / median RT per signed contrast and grouping keys.

    Parameters
    ----------
    by_outcome : bool
        Split by correct / error (choice == sign(contrast)). 0% contrast trials have no
        correct answer and are excluded in this mode.

    Returns columns: *by, [correct,] contrast, n, mean_rt, sem_rt, median_rt, q25_rt, q75_rt.
    """
    require_columns(df)
    d = biased_trials(df) if "prior" in by else df
    keys = list(by)
    if by_outcome:
        d = d[d["contrast"] != 0].assign(correct=lambda x: np.sign(x["contrast"]) == x["choice"])
        keys = keys + ["correct"]
    out = (
        d.groupby(keys + ["contrast"], observed=True)["rt"]
        .agg(
            n="size",
            mean_rt="mean",
            sem_rt=lambda s: sem(s.to_numpy()),
            median_rt="median",
            q25_rt=lambda s: s.quantile(0.25),
            q75_rt=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    return out


def rt_quantile_table(
    df: pd.DataFrame,
    quantiles: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    by: Sequence[str] = ("subject_id", "contrast"),
) -> pd.DataFrame:
    """RT quantiles per grouping (default: per mouse and contrast), as used by the QMP objective."""
    require_columns(df)
    rows = []
    for key, g in df.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n=len(g))
        row.update({f"q{int(round(q * 100)):02d}": float(g["rt"].quantile(q)) for q in quantiles})
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_by_rt_bin(df: pd.DataFrame, n_bins: int = 5, by: Sequence[str] = ("subject_id",)) -> pd.DataFrame:
    """
    Conditional accuracy function: accuracy (non-zero contrasts) per within-group RT quantile bin.

    Returns columns: *by, bin (0 = fastest), rt_median, n, accuracy.
    """
    require_columns(df)
    d = df[df["contrast"] != 0].copy()
    d["correct"] = (np.sign(d["contrast"]) == d["choice"]).astype(float)
    rows = []
    for key, g in d.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        edges = np.quantile(g["rt"].to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, g["rt"].to_numpy(), side="right")
        for b in range(n_bins):
            sub = g[bins == b]
            rows.append(dict(zip(by, key), bin=b, rt_median=float(sub["rt"].median()) if len(sub) else np.nan,
                             n=len(sub), accuracy=float(sub["correct"].mean()) if len(sub) else np.nan))
    return pd.DataFrame(rows)


def across_subject(table: pd.DataFrame, value: str, keys: Sequence[str]) -> pd.DataFrame:
    """Cohort mean and SEM of `value` across mice (each mouse weighted equally) per `keys`."""
    out = (
        table.groupby(list(keys), observed=True)[value]
        .agg(mean="mean", sem=lambda s: sem(s.to_numpy()), n_subjects="count")
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Psychometric fitting
# ---------------------------------------------------------------------------
def _nll(theta, c, n, k) -> float:
    p = np.clip(psychometric_function(c, *theta), 1e-9, 1 - 1e-9)
    return float(-np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))


def _fit_counts(c, n, k, extra_start=None, warm_only=False) -> Dict[str, float]:
    """ML fit from several starts; `warm_only` uses just `extra_start` (bootstrap refits)."""
    if warm_only and extra_start is not None:
        starts = [extra_start]
    else:
        starts = list(_STARTS) + ([extra_start] if extra_start is not None else [])
    best = None
    for s in starts:
        res = minimize(_nll, s, args=(c, n, k), method="L-BFGS-B", bounds=_BOUNDS)
        if best is None or res.fun < best.fun:
            best = res
    return {"mu": best.x[0], "sigma": best.x[1], "gamma": best.x[2], "lam": best.x[3],
            "nll": float(best.fun), "success": bool(best.success)}


def fit_psychometric(
    df: pd.DataFrame,
    n_boot: int = 0,
    seed: int = 0,
    return_boot: bool = False,
) -> Dict[str, object]:
    """
    Maximum-likelihood fit of the lapse-corrected logistic to P(Right) vs signed contrast.

    The data in `df` are pooled as given (filter to one mouse / one block beforehand).
    With `n_boot > 0`, parametric bootstrap 95% intervals (`<param>_lo`, `<param>_hi`) are added.

    Returns
    -------
    dict
        mu, sigma, gamma, lam, nll, success, n_trials, n_contrasts (+ CIs / `boot` array).
    """
    require_columns(df)
    g = df.assign(_right=(df["choice"] == 1).astype(int)).groupby("contrast")["_right"].agg(["size", "sum"])
    c = g.index.to_numpy(dtype=float)
    n = g["size"].to_numpy(dtype=float)
    k = g["sum"].to_numpy(dtype=float)
    nan = {"mu": np.nan, "sigma": np.nan, "gamma": np.nan, "lam": np.nan, "nll": np.nan, "success": False}
    if len(c) < 3:
        out = dict(nan, n_trials=int(n.sum()), n_contrasts=len(c))
        return out
    fit = _fit_counts(c, n, k)
    out: Dict[str, object] = dict(fit, n_trials=int(n.sum()), n_contrasts=len(c))

    if n_boot > 0:
        rng = np.random.default_rng(seed)
        start = (fit["mu"], fit["sigma"], fit["gamma"], fit["lam"])
        boot = np.empty((n_boot, 4))
        for b in range(n_boot):
            kb = rng.binomial(n.astype(int), k / n).astype(float)
            r = _fit_counts(c, n, kb, extra_start=start, warm_only=True)
            boot[b] = [r["mu"], r["sigma"], r["gamma"], r["lam"]]
        for i, name in enumerate(("mu", "sigma", "gamma", "lam")):
            out[f"{name}_lo"], out[f"{name}_hi"] = np.percentile(boot[:, i], [2.5, 97.5])
        if return_boot:
            out["boot"] = boot
    return out


def fit_psychometric_by(
    df: pd.DataFrame,
    by: Sequence[str] = ("subject_id", "prior"),
    n_boot: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run `fit_psychometric` for every group (default: each mouse x block prior)."""
    require_columns(df)
    d = biased_trials(df) if "prior" in by else df
    rows = []
    for key, g in d.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), **fit_psychometric(g, n_boot=n_boot, seed=seed)))
    return pd.DataFrame(rows)


def _count_arrays(g: pd.DataFrame, contrasts: np.ndarray, codes: np.ndarray, n_clusters: int):
    """Per-cluster counts: n[cluster, contrast] and k (Right choices) for the trials of `g`."""
    ci = np.searchsorted(contrasts, g["contrast"].to_numpy(dtype=float))
    flat = codes * len(contrasts) + ci
    size = n_clusters * len(contrasts)
    n = np.bincount(flat, minlength=size).reshape(n_clusters, len(contrasts)).astype(float)
    k = np.bincount(flat, weights=(g["choice"].to_numpy() == 1).astype(float),
                    minlength=size).reshape(n_clusters, len(contrasts))
    return n, k


def _cluster_boot_mu(g: pd.DataFrame, cluster_col: str, n_boot: int, seed: int, starts: Dict[float, tuple]):
    """
    Cluster (e.g. session) bootstrap of the block-specific PSEs: whole clusters are resampled with
    replacement (the same draw for both blocks), so within-session dependence widens the intervals.
    Returns an array (n_boot, 2) of (mu_right_block, mu_left_block).
    """
    codes, _ = pd.factorize(g[cluster_col])
    nc = int(codes.max()) + 1
    contrasts = np.sort(g["contrast"].unique().astype(float))
    counts = {p: _count_arrays(g[g["prior"] == p], contrasts, codes[(g["prior"] == p).to_numpy()], nc)
              for p in (1.0, -1.0)}
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        draw = rng.integers(0, nc, nc)
        for j, p in enumerate((1.0, -1.0)):
            n = counts[p][0][draw].sum(axis=0)
            k = counts[p][1][draw].sum(axis=0)
            ok = n > 0
            out[b, j] = _fit_counts(contrasts[ok], n[ok], k[ok], extra_start=starts[p], warm_only=True)["mu"]
    return out


def prior_shift(
    df: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 0,
    cluster_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Block-prior shift of the psychometric curve for each mouse.

    Fits each mouse's right-biased (prior = +1) and left-biased (prior = -1) blocks separately.
    `shift = mu_left_block - mu_right_block` is in contrast units and positive when the mouse
    behaves as if the stimulus were more rightward in the right-biased block, i.e. a bias
    toward the block's likelier side. The per-block lapse parameters are returned too
    (`gamma_*_block`, `lam_*_block`), because the prior can also act on the asymptotes.

    Bootstrap intervals come from paired resampling of the choice counts (binomial, trial
    level). With `cluster_col` (e.g. 'session_eid') whole sessions are resampled instead, which
    respects the within-session dependence of trials and gives wider, more honest intervals.
    """
    require_columns(df)
    if cluster_col is not None:
        require_columns(df, (cluster_col,))
    rows = []
    for subj, g in biased_trials(df).groupby("subject_id", observed=True):
        right = fit_psychometric(g[g["prior"] > 0], n_boot=0 if cluster_col else n_boot, seed=seed, return_boot=True)
        left = fit_psychometric(g[g["prior"] < 0], n_boot=0 if cluster_col else n_boot, seed=seed + 1, return_boot=True)
        row = {"subject_id": subj, "mu_right_block": right["mu"], "mu_left_block": left["mu"],
               "shift": left["mu"] - right["mu"],
               "sigma_right_block": right["sigma"], "sigma_left_block": left["sigma"],
               "gamma_right_block": right["gamma"], "gamma_left_block": left["gamma"],
               "lam_right_block": right["lam"], "lam_left_block": left["lam"]}
        if n_boot > 0 and cluster_col is not None and np.isfinite(right["mu"]) and np.isfinite(left["mu"]):
            starts = {1.0: (right["mu"], right["sigma"], right["gamma"], right["lam"]),
                      -1.0: (left["mu"], left["sigma"], left["gamma"], left["lam"])}
            boot = _cluster_boot_mu(g, cluster_col, n_boot, seed, starts)
            row["shift_lo"], row["shift_hi"] = np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5])
        elif n_boot > 0 and "boot" in right and "boot" in left:
            diff = left["boot"][:, 0] - right["boot"][:, 0]
            row["shift_lo"], row["shift_hi"] = np.percentile(diff, [2.5, 97.5])
        c0 = g[g["contrast"] == 0]
        p_r = (c0.loc[c0["prior"] > 0, "choice"] == 1).mean()
        p_l = (c0.loc[c0["prior"] < 0, "choice"] == 1).mean()
        row["p_right_c0_right_block"] = p_r
        row["p_right_c0_left_block"] = p_l
        row["prior_effect_c0"] = p_r - p_l
        rows.append(row)
    return pd.DataFrame(rows)
