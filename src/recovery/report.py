"""
Consolidated Wilson & Collins (2019) recovery report: checks the plan's Week 2 acceptance criteria
against the CSVs written by `run_parameter_recovery` / `run_model_recovery`.

    Rule 4 (parameters): Pearson r > 0.8, slope close to 1, and no parameter cross-talk
                         (error correlations Corr(dθ_i, dθ_j) near 0, in particular z vs v_b)
    Rule 5 (models):     BIC confusion matrix with > 85% recovery of every generating mechanism
"""

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model_recovery import MODELS
from .param_recovery import summarize_recovery

R_CRITERION = 0.8
MODEL_RECOVERY_CRITERION = 0.85
CROSSTALK_THRESHOLD = 0.3        # |corr(error_i, error_j)| above this is flagged as a possible trade-off


def model_recovery_summary(df: pd.DataFrame, narrow_margin: float = 10.0) -> Dict[str, object]:
    """
    Confusion P(fit | true), inversion P(true | fit), per-mechanism recovery against the 85% criterion and the
    share of datasets won by less than `narrow_margin` BIC.
    """
    counts = pd.crosstab(df["true_model"], df["winning_model"]).reindex(index=MODELS, columns=MODELS, fill_value=0)
    confusion = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    inversion = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)
    rate = pd.Series({m: confusion.loc[m, m] for m in MODELS})
    bics = df[[f"bic_{m}" for m in MODELS]].to_numpy()
    srt = np.sort(bics, axis=1)
    narrow = (srt[:, 1] - srt[:, 0]) < narrow_margin
    return {
        "counts": counts, "confusion": confusion, "inversion": inversion,
        "recovery_rate": rate, "n_per_model": counts.sum(axis=1),
        "overall": float(np.trace(counts.to_numpy()) / counts.to_numpy().sum()),
        "passes": bool((rate > MODEL_RECOVERY_CRITERION).all()),
        "narrow_share": float(narrow.mean()),
        "narrow_share_by_true": pd.Series(narrow, index=df.index).groupby(df["true_model"]).mean().reindex(MODELS),
    }


EFFECT_PARAM = {"M1": "true_z", "M2": "true_v_b", "M3": "true_b_max"}


def recovery_by_effect_size(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """
    Model-recovery rate of each mechanism by tercile of the size of its own prior effect (|z|, |v_b| or b_max):
    weakly biased datasets cannot be told apart from an unbiased model, so low recovery there is an
    identifiability limit of the design rather than a fitter failure. Also gives the mean BIC margin.
    """
    rows = []
    for m, col in EFFECT_PARAM.items():
        sub = df[df["true_model"] == m]
        if col not in sub or len(sub) < n_bins:
            continue
        eff = sub[col].abs()
        for q, grp in sub.groupby(pd.qcut(eff, n_bins, duplicates="drop"), observed=True):
            ok = grp["winning_model"] == m
            rows.append({"true_model": m, "effect_range": f"{q.left:.2f}-{q.right:.2f}", "n": len(grp),
                         "recovered": float(ok.mean()),
                         "mean_margin_bic": float((grp[[f"bic_{x}" for x in MODELS]].apply(lambda r: np.sort(r.to_numpy())[1] - np.sort(r.to_numpy())[0], axis=1)).mean())})
    return pd.DataFrame(rows)


def error_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of Corr(fit_i - true_i, fit_j - true_j) per trial count for one model's recovery table."""
    names = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    rows = []
    for n, sub in df.groupby("n_trials"):
        err = pd.DataFrame({p: sub[f"fit_{p}"] - sub[f"true_{p}"] for p in names})
        corr = err.corr()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append({"n_trials": n, "param_a": a, "param_b": b, "error_corr": float(corr.loc[a, b])})
    return pd.DataFrame(rows)


def load_parameter_recovery(recovery_dir: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(recovery_dir, "param_recovery_*.csv"))):
        name = os.path.basename(path)[len("param_recovery_"):-len(".csv")]
        if name.endswith("_summary") or name.endswith(".partial"):
            continue
        out[name] = pd.read_csv(path)
    return out


def _md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    f = lambda v: format(v, fmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def build_report(recovery_dir: str = "results/recovery") -> str:
    """Markdown report of the plan-scale recovery (whatever CSVs exist in `recovery_dir`)."""
    lines = ["# Plan-scale recovery report (Wilson & Collins 2019, Rules 4 and 5)", ""]
    mr_path = os.path.join(recovery_dir, "model_recovery_bics.csv")
    if os.path.exists(mr_path):
        mr = pd.read_csv(mr_path)
        s = model_recovery_summary(mr)
        lines += [f"## Model recovery (Rule 5): {len(mr)} datasets (1,000 trials each, the pipeline default)", "",
                  "BIC confusion matrix P(fitted | true), counts:", "", _md(s["counts"].reset_index(), ".0f"), "",
                  "P(fitted | true):", "", _md(s["confusion"].reset_index(), ".1%"), "",
                  "Inversion P(true | fitted):", "", _md(s["inversion"].reset_index(), ".1%"), "",
                  f"Recovery per mechanism: " + ", ".join(f"{m} {r:.0%}" for m, r in s["recovery_rate"].items())
                  + f"; overall {s['overall']:.0%}. **Criterion (> 85% for every mechanism): "
                  + ("PASSED" if s["passes"] else "NOT MET") + ".**", "",
                  f"Datasets won by less than 10 BIC: {s['narrow_share']:.0%} overall "
                  + "(" + ", ".join(f"{m} {v:.0%}" for m, v in s["narrow_share_by_true"].items()) + ").", ""]
        eff = recovery_by_effect_size(mr)
        if len(eff):
            lines += ["Recovery by size of the generating prior effect (terciles of |z|, |v_b|, b_max):", "", _md(eff, ".2f"), ""]
    pr = load_parameter_recovery(recovery_dir)
    if pr:
        lines += ["## Parameter recovery (Rule 4)", "",
                  f"Pearson r of recovered vs true parameter (criterion r > {R_CRITERION}); `*` marks r <= {R_CRITERION}.", ""]
        for name, df in pr.items():
            summ = summarize_recovery(df)
            wide = summ.pivot(index="param", columns="n_trials", values="r")
            flagged = wide.copy().astype(object)
            for c in wide.columns:
                flagged[c] = [f"{v:.2f}{'' if v > R_CRITERION else '*'}" for v in wide[c]]
            slope = summ.pivot(index="param", columns="n_trials", values="slope")
            n_vec = df.groupby("n_trials").size().to_dict()
            lines += [f"### {name} ({n_vec} parameter vectors per N)", "", "Pearson r:", "",
                      _md(flagged.reset_index().rename(columns={"param": "param \\ N"})), "",
                      "Slope of recovered on true (1 = unbiased):", "",
                      _md(slope.reset_index().rename(columns={"param": "param \\ N"}), ".2f"), ""]
            ec = error_correlations(df)
            if len(ec):
                worst = ec.loc[ec["error_corr"].abs().groupby(ec["n_trials"]).idxmax()]
                lines += ["Largest error correlation per N (cross-talk check, |r| > "
                          f"{CROSSTALK_THRESHOLD} flagged):", "", _md(worst.reset_index(drop=True)), ""]
                if {"z", "v_b"} <= set(ec["param_a"]) | set(ec["param_b"]):
                    zv = ec[((ec.param_a == "z") & (ec.param_b == "v_b")) | ((ec.param_a == "v_b") & (ec.param_b == "z"))]
                    lines += ["**z vs v_b error correlation (the plan's cross-talk test):** "
                              + ", ".join(f"N={int(r.n_trials)}: {r.error_corr:+.2f}" for r in zv.itertuples()), ""]
    if len(lines) == 2:
        lines.append("No recovery CSVs found.")
    return "\n".join(lines)
