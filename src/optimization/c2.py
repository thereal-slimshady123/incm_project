"""
C2 (urgency / boundary dynamics) analysis, Week 3, Track B: turns per-mouse fits with FIXED and Weibull-COLLAPSING
bounds into (i) the fixed-vs-collapsing model comparison at the 2.5 s and 5.0 s RT cutoffs, (ii) the invariance test of
the C1 conclusion under collapsing bounds and (iii) the RT-tail evidence for or against urgency.

Everything here works on plain DataFrames (the output of `fit_cohort` and §3.1 trial tables), so it is testable
without running any fits. The statistics are those of C1 (`src.optimization.c1`) and the sign convention is the same:
delta = criterion(first) - criterion(second), NEGATIVE = the first model is better.

Design decisions (details in README, "Week 3, C2")
--------------------------------------------------
* Bounds. Fixed: x in (-a, a). Weibull: a(t) = a exp(-(t/lam)^gamma) (floor 0.05), two extra parameters. The default
  lam range of Track A (0.4-3.5) cannot approach a fixed bound, so C2 fits use lam in (0.4, 20) (`COLLAPSE_BOUNDS`);
  then the fixed bound is the limit lam -> infinity and the models are nested. Weibull fits are warm-started from the
  same mouse's fixed solution (`COLLAPSE_STARTS`) so that optimizer noise cannot make the bigger model look worse.
* Likelihood. The cohort tables are RT-filtered (0.1 s <= RT <= cutoff; 6-23% of the responses of a mouse are slower
  than 2.5 s and 2-9% slower than 5 s). The C1 likelihood puts all simulated mass beyond the cutoff into the last
  QMP bin, which forces a heavy-tailed model to squash its tail and so favours collapsing bounds. The C2 main fits use the
  truncation-aware likelihood (`fit_mouse(truncate=True)`): the model is scored on its RT distribution conditional on the
  window. The C1-likelihood version (`c1like_2p5`) is kept for the invariance test because its fixed-bound fits ARE
  the C1 fits, and as a sensitivity analysis. BIC is only comparable within one (likelihood, cutoff) variant; DELTA-BIC
  between bounds can be compared across variants.
* Decisive evidence. Same rule as C1: |delta BIC| must exceed max(10, 2.8 x pooled run-to-run SD of the replicate fits).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..empirical import tails
from ..empirical.cbf import cbf_slope  # noqa: F401  (re-exported for notebooks)
from ..models.boundaries import weibull_boundary
from . import c1
from .empirical import COLLAPSE_BOUNDS, fitted_params, simulate_from_fit

BOUNDS = ("fixed", "weibull")
BOUND_LABELS = {"fixed": "Fixed bound", "weibull": "Weibull collapsing bound"}
SOURCE_ORDER = ("data", "fixed", "weibull")
BOUND_PAIR: Tuple[Tuple[str, str], ...] = (("weibull", "fixed"),)


# --------------------------------------------------------------------------------------------
# Boundary shape
# --------------------------------------------------------------------------------------------

def bound_fraction(t, lam: float, gamma: float, a: float = 1.0):
    """a(t) / a of the Weibull bound (the same function the simulator uses, including its 0.05 floor)."""
    return weibull_boundary(np.asarray(t, dtype=float), a=a, lam=lam, gamma=gamma) / a


def boundary_curves(best: pd.DataFrame, model: str = "M2", rt_max: float = 2.5,
                    t_grid: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Fitted Weibull bound a(t)/a for every mouse (long: subject_id, t, fraction, lam, gamma)."""
    t = np.linspace(0.0, rt_max, 101) if t_grid is None else np.asarray(t_grid, dtype=float)
    rows = []
    d = best[(best["model"] == model) & (best["bound"] == "weibull") & (best["rt_max"] == rt_max)]
    for _, r in d.iterrows():
        f = bound_fraction(t, r["p_lam"], r["p_gamma"], r["p_a"])
        rows.append(pd.DataFrame({"subject_id": r["subject_id"], "t": t, "fraction": f, "lam": r["p_lam"], "gamma": r["p_gamma"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["subject_id", "t", "fraction", "lam", "gamma"])


# --------------------------------------------------------------------------------------------
# Fixed vs collapsing bound
# --------------------------------------------------------------------------------------------

def bound_table(best: pd.DataFrame, model: str = "M2", rt_max: float = 2.5, metric: str = "bic",
                threshold: float = c1.DECISIVE_MIN_DELTA, probe_times: Sequence[float] = (1.0, 2.5)) -> pd.DataFrame:
    """
    One row per mouse for one (model, cutoff): criterion of the fixed and the Weibull fit, their difference
    `d<metric>_weibull_vs_fixed` (negative = collapsing bound better), the likelihood difference without the
    penalty `dnll_weibull_vs_fixed` (should be <= ~0 because the fixed bound is nested), the winner, margin and
    evidence label, the fitted lam / gamma, and the fitted bound height a(t)/a at `probe_times` (e.g. frac_t1 =
    bound at 1 s relative to its start). `best` comes from `c1.collapse_replicates`.
    """
    d = best[(best["model"] == model) & (best["rt_max"] == rt_max)]
    missing = {"fixed", "weibull"} - set(d["bound"])
    if missing:
        raise ValueError(f"no {sorted(missing)} fits for {model} at rt_max={rt_max}")
    wide = d.pivot(index="subject_id", columns="bound", values=metric)
    nll = d.pivot(index="subject_id", columns="bound", values="nll")
    out = pd.DataFrame(index=wide.index)
    out[f"{metric}_fixed"], out[f"{metric}_weibull"] = wide["fixed"], wide["weibull"]
    out[f"d{metric}_weibull_vs_fixed"] = wide["weibull"] - wide["fixed"]
    out["dnll_weibull_vs_fixed"] = nll["weibull"] - nll["fixed"]
    out["winner"] = np.where(out[f"d{metric}_weibull_vs_fixed"] < 0, "weibull", "fixed")
    out["margin"] = out[f"d{metric}_weibull_vs_fixed"].abs()
    out["evidence"] = np.where(out["margin"] > threshold, "decisive", "inconclusive")
    out["winner_decisive"] = np.where(out["evidence"] == "decisive", out["winner"], "none")
    w = d[d["bound"] == "weibull"].set_index("subject_id")
    for k in ("p_lam", "p_gamma"):
        out[k] = w[k]
    for t in probe_times:
        out[f"frac_t{t:g}"] = [float(bound_fraction(t, r.p_lam, r.p_gamma)) for r in out.itertuples()]
    lam_hi = COLLAPSE_BOUNDS["lam"][1]
    out["lam_at_ceiling"] = out["p_lam"] >= 0.98 * lam_hi
    for b in BOUNDS:
        sub = d[d["bound"] == b].set_index("subject_id")
        out[f"bic_spread_{b}"] = sub["bic_spread"] if "bic_spread" in sub else np.nan
    return out.reset_index()


def bound_tests(table: pd.DataFrame, metric: str = "bic", n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Across-mouse tests of 'Weibull vs fixed' (the C1 test battery; one pair, so no Holm correction needed)."""
    return c1.pairwise_tests(table, metric, n_boot=n_boot, seed=seed, pairs=BOUND_PAIR)


def cutoff_consistency(tab_a: pd.DataFrame, tab_b: pd.DataFrame, metric: str = "bic") -> Dict[str, float]:
    """
    Agreement of the per-mouse delta-BIC (Weibull - fixed) between two variants/cutoffs (e.g. 2.5 s vs 5.0 s).
    BICs are not comparable across data sets, but the model differences are. Returns the Spearman rank
    correlation over mice, the number of mice with the same sign and the two group means.
    """
    col = f"d{metric}_weibull_vs_fixed"
    m = tab_a[["subject_id", col]].merge(tab_b[["subject_id", col]], on="subject_id", suffixes=("_a", "_b"))
    rho = stats.spearmanr(m[col + "_a"], m[col + "_b"]) if len(m) > 2 else None
    return {"n_mice": len(m), "spearman_rho": float(rho.statistic) if rho is not None else float("nan"),
            "spearman_p": float(rho.pvalue) if rho is not None else float("nan"),
            "n_same_sign": int((np.sign(m[col + "_a"]) == np.sign(m[col + "_b"])).sum()),
            "mean_a": float(m[col + "_a"].mean()), "mean_b": float(m[col + "_b"].mean())}


# --------------------------------------------------------------------------------------------
# Invariance of the C1 conclusion under collapsing bounds
# --------------------------------------------------------------------------------------------

def invariance_table(best: pd.DataFrame, rt_max: float = 2.5, metric: str = "bic",
                     threshold: float = c1.DECISIVE_MIN_DELTA) -> pd.DataFrame:
    """
    Per mouse: the C1 comparison (M1 / M2 / M3, `c1.bic_table`) under the fixed and under the Weibull bound side by side,
    suffixes `_fixed` / `_weibull`: criteria, pairwise deltas, winner and evidence, plus `shift_<pair>` = delta under
    Weibull minus delta under fixed (how much the collapsing bound changed that model contrast) and `same_winner`.
    """
    tabs = {}
    for b in BOUNDS:
        have = set(best[(best["bound"] == b) & (best["rt_max"] == rt_max)]["model"])
        if not set(c1.MODELS) <= have:
            raise ValueError(f"need {c1.MODELS} fitted with the {b} bound at rt_max={rt_max}; have {sorted(have)}")
        tabs[b] = c1.bic_table(best, metric, rt_max=rt_max, bound=b, threshold=threshold).set_index("subject_id")
    cols = [f"{metric}_{m}" for m in c1.MODELS] + [f"d{metric}_{a}_vs_{b}" for a, b in c1.PAIRS] + ["winner", "margin", "evidence"]
    out = pd.concat({b: tabs[b][cols] for b in BOUNDS}, axis=1)
    out.columns = [f"{c}_{b}" for b, c in out.columns]
    for a, b in c1.PAIRS:
        name = f"d{metric}_{a}_vs_{b}"
        out[f"shift_{a}_vs_{b}"] = out[f"{name}_weibull"] - out[f"{name}_fixed"]
    out["same_winner"] = out["winner_fixed"] == out["winner_weibull"]
    return out.reset_index()


def invariance_tests(best: pd.DataFrame, inv: pd.DataFrame, rt_max: float = 2.5, metric: str = "bic",
                     n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """
    The C1 across-mouse tests repeated under each bound (rows with `bound` = fixed / weibull), plus for every model pair
    a `bound` = 'shift' row testing whether the pair's delta-BIC DIFFERS between the bounds (paired Wilcoxon / t over mice
    on `shift_<pair>`): the direct test of invariance. `mean_delta` of a shift row is in BIC units; `rel_mean`, `rel_ci_lo`,
    `rel_ci_hi` express the shift and its bootstrap CI as a fraction of the mean effect under the fixed bound (NaN if that effect
    is below 1 BIC), the equivalence-style reading: invariance is supported if the whole CI is a small fraction of the effect.
    A non-significant shift alone is weak evidence of invariance with 10 mice and ~250 BIC of optimizer noise per fit.
    """
    parts = []
    for b in BOUNDS:
        tab = c1.bic_table(best, metric, rt_max=rt_max, bound=b)
        parts.append(c1.pairwise_tests(tab, metric, n_boot=n_boot, seed=seed).assign(bound=b))
    fixed_mean = parts[0].set_index("pair")["mean_delta"]          # parts[0] is the fixed-bound table
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in c1.PAIRS:
        x = inv[f"shift_{a}_vs_{b}"].to_numpy(dtype=float)
        n = len(x)
        boots = x[rng.integers(0, n, (n_boot, n))].mean(axis=1)
        try:
            w_p = float(stats.wilcoxon(x).pvalue)
        except ValueError:
            w_p = float("nan")
        rows.append({"pair": f"{a} vs {b}", "bound": "shift", "n_mice": n, "mean_delta": float(x.mean()),
                     "median_delta": float(np.median(x)), "ci_lo": float(np.percentile(boots, 2.5)),
                     "ci_hi": float(np.percentile(boots, 97.5)), "n_first_better": int((x < 0).sum()),
                     "n_second_better": int((x > 0).sum()), "wilcoxon_p": w_p,
                     "ttest_p": float(stats.ttest_1samp(x, 0.0).pvalue) if n > 2 else float("nan")})
        scale = abs(float(fixed_mean[f"{a} vs {b}"]))
        for key, val in (("rel_mean", rows[-1]["mean_delta"]), ("rel_ci_lo", rows[-1]["ci_lo"]), ("rel_ci_hi", rows[-1]["ci_hi"])):
            rows[-1][key] = val / scale if scale > 1.0 else float("nan")
    return pd.concat(parts + [pd.DataFrame(rows)], ignore_index=True)


def invariance_correlations(inv: pd.DataFrame, metric: str = "bic") -> pd.DataFrame:
    """Per model pair: Spearman correlation across mice of the fixed-bound delta with the Weibull-bound delta."""
    rows = []
    for a, b in c1.PAIRS:
        name = f"d{metric}_{a}_vs_{b}"
        r = stats.spearmanr(inv[f"{name}_fixed"], inv[f"{name}_weibull"])
        rows.append({"pair": f"{a} vs {b}", "spearman_rho": float(r.statistic), "p": float(r.pvalue),
                     "same_sign": int((np.sign(inv[f"{name}_fixed"]) == np.sign(inv[f"{name}_weibull"])).sum()),
                     "n_mice": len(inv)})
    return pd.DataFrame(rows)


def c1_verdict(tests: pd.DataFrame, cbf_fit: Optional[pd.DataFrame] = None, alpha: float = 0.05) -> Dict[str, object]:
    """
    The C1 conclusion in three comparable statements, from `c1.pairwise_tests` output (Holm-adjusted Wilcoxon) and
    optionally the CBF fit table:
      m1_rejected : M1 worse than both M2 and M3 across mice (mean delta > 0, Holm p < alpha for both pairs)
      m2_vs_m3    : 'M2' / 'M3' if one is significantly better by BIC, else 'not separable'
      cbf_closest : model whose predicted CBF has the smallest RMSE in the most mice
    Two analyses (e.g. fixed vs Weibull) are *invariant* if the three statements are identical.
    """
    t = tests.set_index("pair")
    sig = lambda pair: bool(t.loc[pair, "wilcoxon_p_holm"] < alpha)
    m1 = (t.loc["M1 vs M2", "mean_delta"] > 0 and sig("M1 vs M2")) and (t.loc["M1 vs M3", "mean_delta"] > 0 and sig("M1 vs M3"))
    if sig("M3 vs M2"):
        m2m3 = "M3" if t.loc["M3 vs M2", "mean_delta"] < 0 else "M2"
    else:
        m2m3 = "not separable"
    out: Dict[str, object] = {"m1_rejected": bool(m1), "m2_vs_m3": m2m3}
    if cbf_fit is not None:
        idx = cbf_fit.reset_index(drop=True).groupby("subject_id")["rmse"].idxmin()
        out["cbf_closest"] = str(cbf_fit.reset_index(drop=True).loc[idx, "model"].value_counts().idxmax())
    return out


# --------------------------------------------------------------------------------------------
# RT tail: data vs fixed vs collapsing
# --------------------------------------------------------------------------------------------

def tail_tables(
    cohort: pd.DataFrame,
    best: pd.DataFrame,
    model: str = "M2",
    rt_max: float = 5.0,
    bounds: Sequence[str] = BOUNDS,
    n_rep: int = 20,
    seed: int = 0,
    survival_grid: Sequence[float] = tails.DEFAULT_SURVIVAL_GRID,
    hazard_edges: Sequence[float] = tails.DEFAULT_HAZARD_EDGES,
    accuracy_edges: Sequence[float] = tails.DEFAULT_ACCURACY_EDGES,
) -> Dict[str, pd.DataFrame]:
    """
    Survival, hazard, accuracy-by-RT-window and the one-row summary of every mouse's REAL RTs (source 'data') and of the
    posterior-predictive RTs of its fitted `model` under each bound (the mouse's own trials replayed `n_rep` times and
    cut to the same window 0.1 s <= RT <= rt_max). Returns {'survival', 'hazard', 'accuracy', 'summary'} long tables
    with a `source` column.
    """
    out: Dict[str, list] = {k: [] for k in ("survival", "hazard", "accuracy", "summary")}
    fits = best[(best["model"] == model) & (best["rt_max"] == rt_max)]
    for subj, trials in cohort.groupby("subject_id", observed=True):
        frames = {"data": trials}
        for b in bounds:
            r = fits[(fits["subject_id"] == subj) & (fits["bound"] == b)]
            if len(r):
                frames[b] = simulate_from_fit(trials, fitted_params(r.iloc[0]), model, b, rt_max, n_rep=n_rep, seed=seed)
        for src, df in frames.items():
            df = df.assign(subject_id=subj)
            for name, tab in (("survival", tails.survival_table(df, survival_grid)), ("hazard", tails.hazard_table(df, hazard_edges)),
                              ("accuracy", tails.accuracy_by_rt_window(df, accuracy_edges)), ("summary", tails.tail_summary(df))):
                out[name].append(tab.assign(source=src))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def tail_errors(tt: Dict[str, pd.DataFrame], t_min: float = 1.0) -> pd.DataFrame:
    """
    Per (mouse, bound): how far the model's RT tail is from the mouse's, measured three ways
      survival_error  mean |S_data(t) - S_model(t)| over t >= t_min (S = P(RT > t))
      hazard_log_err  |log(hazard ratio model) - log(hazard ratio data)|   (late/early hazard, see tails.late_hazard_ratio)
      accuracy_error  |late-RT accuracy model - data|   (|c| >= 0.25, RT > 2 s)
    plus the signed late/early hazard ratios. Smaller = closer to the data.
    """
    surv, summ = tt["survival"], tt["summary"].set_index(["subject_id", "source"])
    rows = []
    for subj, g in surv.groupby("subject_id"):
        data = g[g["source"] == "data"]
        for b in [s for s in g["source"].unique() if s != "data"]:
            model = g[g["source"] == b]
            d, m = summ.loc[(subj, "data")], summ.loc[(subj, b)]
            rows.append({
                "subject_id": subj, "bound": b, "survival_error": tails.survival_distance(data, model, t_min),
                "hazard_ratio_data": d["hazard_ratio"], "hazard_ratio_model": m["hazard_ratio"],
                "log_hr_diff": np.log(d["hazard_ratio"]) - np.log(m["hazard_ratio"]) if d["hazard_ratio"] > 0 and m["hazard_ratio"] > 0 else np.nan,
                "hazard_log_err": abs(np.log(m["hazard_ratio"]) - np.log(d["hazard_ratio"])) if d["hazard_ratio"] > 0 and m["hazard_ratio"] > 0 else np.nan,
                "p_gt2p5_data": d["p_gt_2.5"], "p_gt2p5_model": m["p_gt_2.5"],
                "accuracy_gt2s_data": d["accuracy_gt2s"], "accuracy_gt2s_model": m["accuracy_gt2s"],
                "accuracy_error": abs(m["accuracy_gt2s"] - d["accuracy_gt2s"]),
                "accuracy_diff": d["accuracy_gt2s"] - m["accuracy_gt2s"], "p_gt2p5_diff": d["p_gt_2.5"] - m["p_gt_2.5"],
            })
    return pd.DataFrame(rows)


def tail_error_tests(errors: pd.DataFrame, columns: Sequence[str] = ("survival_error", "hazard_log_err", "accuracy_error")) -> pd.DataFrame:
    """Paired across-mouse Wilcoxon / t tests of the tail errors: Weibull minus fixed (negative = collapsing closer to the data)."""
    rows = []
    for col in columns:
        w = errors.pivot(index="subject_id", columns="bound", values=col).dropna()
        if len(w) < 3 or not {"fixed", "weibull"} <= set(w.columns):
            continue
        diff = (w["weibull"] - w["fixed"]).to_numpy()
        try:
            wp = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wp = float("nan")
        rows.append({"metric": col, "n_mice": len(w), "mean_fixed": float(w["fixed"].mean()), "mean_weibull": float(w["weibull"].mean()),
                     "mean_delta": float(diff.mean()), "n_weibull_closer": int((diff < 0).sum()), "wilcoxon_p": wp,
                     "ttest_p": float(stats.ttest_rel(w["weibull"], w["fixed"]).pvalue)})
    return pd.DataFrame(rows)


def tail_signed_tests(errors: pd.DataFrame, bound: str = "fixed",
                      columns: Sequence[str] = ("log_hr_diff", "accuracy_diff", "p_gt2p5_diff")) -> pd.DataFrame:
    """
    Signed data-minus-model differences of the tail statistics for one bound, tested against 0 across mice (Wilcoxon, t):
      log_hr_diff    log(late/early hazard ratio) data - model   (> 0: the mice's late hazard rises MORE than the model's)
      accuracy_diff  accuracy for RT > 2 s, data - model          (> 0: the mice stay more accurate when slow than the model)
      p_gt2p5_diff   fraction of RTs > 2.5 s, data - model        (> 0: the mice have a heavier slow tail)
    This is the fair test of the tail: data and model RTs are cut to the same window, so truncation affects both equally.
    """
    d = errors[errors["bound"] == bound]
    rows = []
    for col in columns:
        x = d[col].dropna().to_numpy()
        if len(x) < 3:
            continue
        try:
            wp = float(stats.wilcoxon(x).pvalue)
        except ValueError:
            wp = float("nan")
        rows.append({"bound": bound, "metric": col, "n_mice": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                     "n_positive": int((x > 0).sum()), "wilcoxon_p": wp, "ttest_p": float(stats.ttest_1samp(x, 0.0).pvalue)})
    return pd.DataFrame(rows)


def empirical_tail_tests(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive model-free tests on the mice themselves (rows with source 'data'): is the late/early hazard ratio different
    from 1 (log ratio vs 0, Wilcoxon), and does accuracy for slow responses (> 2 s) fall below accuracy for fast ones (< 1 s)?
    CAUTION: the RTs are truncated at the cutoff, which inflates the hazard of late windows (conditional on RT <= 5 s the hazard
    rises towards 5 s even for a constant true hazard), so the hazard test is NOT evidence of urgency by itself; use
    `tail_signed_tests` (data against the fitted model, cut to the same window) for that. The accuracy test is unaffected.
    """
    d = summary[summary["source"] == "data"].dropna(subset=["hazard_ratio"])
    rows = []
    log_ratio = np.log(d["hazard_ratio"].clip(lower=1e-6))
    for name, x in (("log hazard ratio (late/early) vs 0 [truncation-inflated, descriptive]", log_ratio), ("accuracy(<1 s) - accuracy(>2 s) vs 0", d["accuracy_drop"].dropna())):
        try:
            wp = float(stats.wilcoxon(x).pvalue)
        except ValueError:
            wp = float("nan")
        rows.append({"test": name, "n_mice": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
                     "n_positive": int((x > 0).sum()), "wilcoxon_p": wp})
    return pd.DataFrame(rows)
