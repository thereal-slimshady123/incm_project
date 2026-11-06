"""
C1 (prior mechanism) analysis, Week 3: turns per-mouse fits of M1 (starting point z), M2 (drift
bias v_b) and M3 (dynamic bias b(t)) into the model-comparison tables, the across-mouse statistics
and the model-vs-data Conditional Bias Function (CBF) comparison.

Everything here works on plain DataFrames (the output of `fit_cohort` and §3.1 trial tables), so it
is testable without running any fits.

Design decisions
----------------
* Replicate fits. The simulated QMP likelihood is noisy and Nelder-Mead stops at different local
  optima for different seeds (BIC spread of tens of points on one real mouse). `fit_cohort(seeds=...)`
  therefore runs independent replicates; `collapse_replicates` keeps the best-BIC replicate (a bad
  run can only be too high, never too low, up to Monte-Carlo noise in the final NLL) and records
  the replicate spread, which `noise_floor` pools over all fits into one estimate of the
  between-run SD. A model difference is called *decisive* only if it exceeds both the usual BIC
  threshold of 10 and about 2.8 pooled SDs (95% for a difference of two noisy fits).
* ΔBIC convention: delta = BIC(model) - BIC(reference); NEGATIVE = model beats the reference.
* Tests are across mice (the unit of replication): paired Wilcoxon signed-rank, paired t on the
  per-mouse BICs, an exact sign test, and a bootstrap CI over mice; p-values are Holm-adjusted over
  the three model pairs.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..empirical.cbf import DEFAULT_MAX_ABS_CONTRAST, cbf_slope, compute_cbf
from .empirical import fitted_params, simulate_from_fit

MODELS = ("M1", "M2", "M3")
MODEL_LABELS = {"M1": "M1 start point (z)", "M2": "M2 drift bias (v_b)", "M3": "M3 dynamic bias b(t)",
                "JOINT": "JOINT start point + drift bias"}
MODEL_SHAPE = {"M1": "falling", "M2": "flat", "M3": "rising"}
PAIRS: Tuple[Tuple[str, str], ...] = (("M1", "M2"), ("M3", "M2"), ("M1", "M3"))
KEYS = ["subject_id", "model", "bound", "rt_max"]

# Mice whose behaviour contains block-dependent lapses (errors at 100% contrast that depend on the
# block), which the lapse-free model family cannot reproduce through z / v_b / b(t) (README, Week 3).
LAPSE_AFFECTED = ("KS019", "KS021", "CSH_ZAD_002", "NYU-01")

DECISIVE_MIN_DELTA = 10.0     # conventional "strong" BIC evidence
DECISIVE_SD_MULT = 2.8        # 1.96 * sqrt(2): 95% band for the difference of two fits with equal noise


# --------------------------------------------------------------------------------------------
# Replicates, noise, tables
# --------------------------------------------------------------------------------------------

def collapse_replicates(fits: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (mouse, model, bound, rt_max): the replicate with the lowest BIC, plus
    `n_reps`, `bic_spread` (max - min over replicates; 0 for a single run) and `bic_mean`.
    """
    fits = fits.reset_index(drop=True)
    if "seed" not in fits.columns:
        fits["seed"] = 0
    g = fits.groupby(KEYS, sort=True)
    best = fits.loc[g["bic"].idxmin()].set_index(KEYS)
    best["n_reps"] = g.size()
    best["bic_spread"] = g["bic"].max() - g["bic"].min()
    best["bic_mean"] = g["bic"].mean()
    return best.reset_index()


def noise_floor(fits: pd.DataFrame) -> Dict[str, float]:
    """
    Pooled between-run SD of the fitted BIC from replicate fits of the same (mouse, model).

    With R replicates per fit the unbiased pooled variance is mean over fits of var(bic, ddof=1).
    Returns {'sd', 'n_fits', 'median_spread', 'max_spread'}; sd is NaN without replicates.
    """
    g = fits.groupby(KEYS)["bic"]
    var = g.var(ddof=1).dropna()
    spread = g.max() - g.min()
    return {
        "sd": float(np.sqrt(var.mean())) if len(var) else float("nan"),
        "n_fits": int(len(var)),
        "median_spread": float(spread.median()),
        "max_spread": float(spread.max()),
    }


def decisive_threshold(sd: float) -> float:
    """Smallest |delta BIC| treated as decisive given the pooled run-to-run SD."""
    return float(max(DECISIVE_MIN_DELTA, DECISIVE_SD_MULT * sd)) if np.isfinite(sd) else DECISIVE_MIN_DELTA


def bic_table(best: pd.DataFrame, metric: str = "bic", rt_max: float = 2.5, bound: str = "fixed",
              threshold: float = DECISIVE_MIN_DELTA) -> pd.DataFrame:
    """
    Per-mouse model-comparison table (one row per mouse) from `collapse_replicates` output:
    the criterion of each model, ΔBIC against baseline M2 (negative = better than M2), the pairwise
    M1-vs-M3 difference, the winner, the margin to the runner-up and an evidence label
    ('decisive' if the margin exceeds `threshold`, else 'inconclusive').
    """
    d = best[(best["rt_max"] == rt_max) & (best["bound"] == bound)]
    wide = d.pivot(index="subject_id", columns="model", values=metric)
    wide = wide[[m for m in MODELS if m in wide.columns]]
    out = wide.add_prefix(f"{metric}_").copy()
    for a, b in PAIRS:
        if a in wide and b in wide:
            out[f"d{metric}_{a}_vs_{b}"] = wide[a] - wide[b]
    srt = np.sort(wide.to_numpy(), axis=1)
    out["winner"] = wide.idxmin(axis=1)
    out["margin"] = srt[:, 1] - srt[:, 0]
    out["evidence"] = np.where(out["margin"] > threshold, "decisive", "inconclusive")
    out["winner_decisive"] = np.where(out["evidence"] == "decisive", out["winner"], "none")
    n_par = d.pivot(index="subject_id", columns="model", values="n_params")
    for m in n_par.columns:
        out[f"k_{m}"] = n_par[m]
    return out.reset_index()


def bic_weights(table: pd.DataFrame, metric: str = "bic") -> pd.DataFrame:
    """Schwarz weights exp(-Δ/2) / Σ per mouse (posterior model probabilities under equal priors)."""
    cols = [f"{metric}_{m}" for m in MODELS if f"{metric}_{m}" in table]
    v = table[cols].to_numpy()
    w = np.exp(-(v - v.min(axis=1, keepdims=True)) / 2.0)
    w /= w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, columns=[f"w_{c.split('_')[-1]}" for c in cols], index=table["subject_id"]).reset_index()


# --------------------------------------------------------------------------------------------
# Across-mouse statistics
# --------------------------------------------------------------------------------------------

def _holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_tests(table: pd.DataFrame, metric: str = "bic", n_boot: int = 10000, seed: int = 0,
                   pairs: Sequence[Tuple[str, str]] = PAIRS) -> pd.DataFrame:
    """
    Paired across-mouse tests of every model pair (A vs B) on the per-mouse criterion.
    `mean_delta` = mean over mice of (A - B); negative favours A.

    Columns: n_mice, mean_delta, median_delta, sum_delta (log group Bayes factor x 2 for BIC),
    ci_lo/ci_hi (bootstrap over mice), n_first_better / n_second_better (mice favouring A / B), sign_p (exact two-sided), wilcoxon_p, ttest_p,
    dz (paired Cohen's d) and Holm-adjusted p for the three tests. `pairs` selects the model pairs
    (default M1/M2/M3; ('JOINT', 'M2') etc. need `bic_JOINT` in the table, see `joint_comparison`).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        ca, cb = f"{metric}_{a}", f"{metric}_{b}"
        if ca not in table or cb not in table:
            continue
        diff = (table[ca] - table[cb]).to_numpy(dtype=float)
        n = len(diff)
        boots = diff[rng.integers(0, n, (n_boot, n))].mean(axis=1)
        wins = int((diff < 0).sum())
        n_nz = int((diff != 0).sum())
        try:
            w_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w_p = float("nan")
        t_p = float(stats.ttest_rel(table[ca], table[cb]).pvalue) if n >= 3 else float("nan")
        rows.append({
            "pair": f"{a} vs {b}", "n_mice": n, "mean_delta": float(diff.mean()), "median_delta": float(np.median(diff)),
            "sum_delta": float(diff.sum()), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "n_first_better": wins, "n_second_better": int((diff > 0).sum()),
            "sign_p": float(stats.binomtest(wins, n_nz, 0.5).pvalue) if n_nz else float("nan"),
            "wilcoxon_p": w_p, "ttest_p": t_p,
            "dz": float(diff.mean() / diff.std(ddof=1)) if n > 2 and diff.std(ddof=1) > 0 else float("nan"),
        })
    out = pd.DataFrame(rows)
    for col in ("sign_p", "wilcoxon_p", "ttest_p"):
        ok = out[col].notna()
        out[col + "_holm"] = np.nan
        if ok.any():
            out.loc[ok, col + "_holm"] = _holm(out.loc[ok, col])
    return out


def winner_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Per-model count of mice won (all, and decisive only)."""
    rows = []
    for m in MODELS:
        rows.append({"model": m, "n_best": int((table["winner"] == m).sum()),
                     "n_decisive": int((table["winner_decisive"] == m).sum())})
    rows.append({"model": "none (inconclusive)", "n_best": np.nan, "n_decisive": int((table["winner_decisive"] == "none").sum())})
    return pd.DataFrame(rows)


def subset_summary(table: pd.DataFrame, subsets: Dict[str, Optional[Sequence[str]]], metric: str = "bic") -> pd.DataFrame:
    """
    Pairwise tests repeated on mouse subsets, e.g. {'all': None, 'excluding lapse-affected': LAPSE_AFFECTED}.
    A value is the list of mice to EXCLUDE (None = nobody). Long table with a `subset` column.
    """
    parts = []
    for name, excl in subsets.items():
        t = table if not excl else table[~table["subject_id"].isin(list(excl))]
        if len(t) >= 3:
            parts.append(pairwise_tests(t, metric).assign(subset=name, n_in_subset=len(t)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------------------------
# Model-vs-data CBF
# --------------------------------------------------------------------------------------------

def _edges(tab: pd.DataFrame) -> np.ndarray:
    return np.r_[tab["rt_lo"].to_numpy(), tab["rt_hi"].iloc[-1]]


def empirical_cbf(cohort: pd.DataFrame, n_bins: int = 5, max_abs_contrast: float = DEFAULT_MAX_ABS_CONTRAST) -> pd.DataFrame:
    """Per-mouse empirical CBF (same estimator and defaults as results/empirical/cbf_curves.csv)."""
    parts = []
    for s, g in cohort.groupby("subject_id", observed=True):
        parts.append(compute_cbf(g, n_bins=n_bins, max_abs_contrast=max_abs_contrast).assign(subject_id=s))
    return pd.concat(parts, ignore_index=True)


def predicted_cbf(
    cohort: pd.DataFrame,
    best: pd.DataFrame,
    models: Sequence[str] = MODELS,
    n_rep: int = 20,
    n_bins: int = 5,
    max_abs_contrast: float = DEFAULT_MAX_ABS_CONTRAST,
    rt_max: float = 2.5,
    bound: str = "fixed",
    seed: int = 0,
    emp: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Posterior-predictive CBF of every fitted (mouse, model): the mouse's own trial covariates are
    replayed `n_rep` times through the fitted model and scored on the mouse's EMPIRICAL RT bin edges,
    so predicted and observed bins are directly comparable. Returns a long table
    (subject_id, model, bin, rt_median, cbf, n_trials, n_sim).
    """
    emp = emp if emp is not None else empirical_cbf(cohort, n_bins, max_abs_contrast)
    rows = []
    for _, r in best[(best["bound"] == bound) & (best["rt_max"] == rt_max) & (best["model"].isin(models))].iterrows():
        subj, model = r["subject_id"], r["model"]
        edges = _edges(emp[emp["subject_id"] == subj].sort_values("bin"))
        trials = cohort[cohort["subject_id"] == subj]
        sim = simulate_from_fit(trials, fitted_params(r), model, bound, rt_max, n_rep=n_rep, seed=seed)
        tab = compute_cbf(sim, n_bins=n_bins, max_abs_contrast=max_abs_contrast, bin_edges=edges)
        tab["subject_id"], tab["model"], tab["n_sim"] = subj, model, len(sim)
        rows.append(tab[["subject_id", "model", "bin", "rt_median", "cbf", "n_trials", "n_sim"]])
    return pd.concat(rows, ignore_index=True)


def cbf_fit_table(emp: pd.DataFrame, pred: pd.DataFrame, shapes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Per (mouse, model) agreement between predicted and empirical CBF:
    `rmse` (ΔP(Right) units), `chi2_per_bin` (squared error in units of the empirical
    analytic SE; independent-trial SEs, so descriptive), `slope_pred`, `slope_emp`, `slope_err`.
    `shapes` (optional `cbf_slopes.csv` from Track B) adds the empirical shape label.
    """
    e = emp.set_index(["subject_id", "bin"])
    rows = []
    for (subj, model), g in pred.groupby(["subject_id", "model"]):
        ge = e.loc[subj].loc[g["bin"]]
        err = g["cbf"].to_numpy() - ge["cbf"].to_numpy()
        se = ge["se"].to_numpy()
        ok = np.isfinite(err) & np.isfinite(se) & (se > 0)
        rows.append({
            "subject_id": subj, "model": model,
            "rmse": float(np.sqrt(np.mean(err[np.isfinite(err)] ** 2))),
            "chi2_per_bin": float(np.mean((err[ok] / se[ok]) ** 2)),
            "slope_pred": cbf_slope(g["cbf"]), "slope_emp": cbf_slope(ge["cbf"]),
        })
    out = pd.DataFrame(rows)
    out["slope_err"] = out["slope_pred"] - out["slope_emp"]
    if shapes is not None:
        out = out.merge(shapes[["subject_id", "shape"]].rename(columns={"shape": "emp_shape"}), on="subject_id", how="left")
    return out


def cbf_model_tests(fit_table: pd.DataFrame, value: str = "rmse",
                    pairs: Sequence[Tuple[str, str]] = PAIRS) -> pd.DataFrame:
    """Paired across-mouse Wilcoxon / t tests of the CBF discrepancy `value` for every model pair."""
    wide = fit_table.pivot(index="subject_id", columns="model", values=value)
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).to_numpy()
        try:
            w_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w_p = float("nan")
        rows.append({"pair": f"{a} vs {b}", "metric": value, "mean_delta": float(diff.mean()),
                     "n_first_closer": int((diff < 0).sum()), "n_mice": len(diff),
                     "wilcoxon_p": w_p, "ttest_p": float(stats.ttest_rel(wide[a], wide[b]).pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = _holm(out["wilcoxon_p"].fillna(1.0))
    return out


def concordance(table: pd.DataFrame, emp_shapes: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    """
    Per mouse: BIC winner, empirical CBF shape (Track B), the shape the winner predicts (M1 falling,
    M2 flat, M3 rising) and whether they agree; plus the model whose predicted CBF is closest to
    the data (smallest RMSE) and the winner's own predicted slope.
    """
    out = table[["subject_id", "winner", "evidence"]].merge(
        emp_shapes[["subject_id", "shape", "slope"]].rename(columns={"shape": "emp_shape", "slope": "emp_slope"}), on="subject_id")
    out["winner_shape"] = out["winner"].map(MODEL_SHAPE)
    out["shape_agrees"] = out["winner_shape"] == out["emp_shape"]
    fit_table = fit_table.reset_index(drop=True)
    closest = fit_table.loc[fit_table.groupby("subject_id")["rmse"].idxmin(), ["subject_id", "model"]].rename(columns={"model": "cbf_closest"})
    return out.merge(closest, on="subject_id", how="left")


JOINT_PAIRS: Tuple[Tuple[str, str], ...] = (("JOINT", "M2"), ("JOINT", "M1"), ("JOINT", "M3"))


def joint_comparison(best: pd.DataFrame, joint_best: pd.DataFrame, metric: str = "bic", rt_max: float = 2.5,
                     bound: str = "fixed") -> pd.DataFrame:
    """
    Per-mouse comparison of the combined start-point + drift-bias model (JOINT, one extra parameter) with
    M1/M2/M3: criterion of every model, JOINT minus each of the others (negative = JOINT better) and the
    fitted z and v_b. Both inputs come from `collapse_replicates`.
    """
    both = pd.concat([best, joint_best], ignore_index=True)
    d = both[(both["rt_max"] == rt_max) & (both["bound"] == bound)]
    wide = d.pivot(index="subject_id", columns="model", values=metric).add_prefix(f"{metric}_")
    for m in ("M1", "M2", "M3"):
        wide[f"d{metric}_JOINT_vs_{m}"] = wide[f"{metric}_JOINT"] - wide[f"{metric}_{m}"]
    jp = joint_best[(joint_best["rt_max"] == rt_max) & (joint_best["bound"] == bound)].set_index("subject_id")
    for k in ("p_z", "p_v_b"):
        wide[k] = jp[k]
    wide["winner"] = wide[[f"{metric}_{m}" for m in ("M1", "M2", "M3", "JOINT")]].idxmin(axis=1).str.replace(f"{metric}_", "")
    return wide.reset_index()


def parameter_table(best: pd.DataFrame, rt_max: float = 2.5, bound: str = "fixed") -> pd.DataFrame:
    """Fitted parameters of every (mouse, model) in the long format, for the supplementary table."""
    d = best[(best["rt_max"] == rt_max) & (best["bound"] == bound)]
    cols = ["subject_id", "model", "n_trials", "n_params", "nll", "bic", "bic_spread", "n_reps", "success"]
    cols += sorted(c for c in d.columns if c.startswith("p_"))
    return d[cols].sort_values(["subject_id", "model"]).reset_index(drop=True)


def at_bound(best: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """
    Parameters sitting at (within `tol` of the range of) their search bounds. Such fits are
    effectively constrained, and an M3 `b_max` at 0 or an `s_*` at 0 is informative, not a failure.
    """
    from .fit import PARAM_SPECS
    rows = []
    for _, r in best.iterrows():
        for k, v in fitted_params(r).items():
            if k not in PARAM_SPECS or k == "tau_b":
                continue
            lo, hi = PARAM_SPECS[k]["bounds"]
            w = hi - lo
            if v <= lo + tol * w or v >= hi - tol * w:
                rows.append({"subject_id": r["subject_id"], "model": r["model"], "param": k, "value": v,
                             "bound": "lower" if v <= lo + tol * w else "upper"})
    return pd.DataFrame(rows, columns=["subject_id", "model", "param", "value", "bound"])
