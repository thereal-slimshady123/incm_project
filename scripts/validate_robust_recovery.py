"""
Model recovery for the ROBUST C1 model class (Wilson & Collins 2019, rule 5), at the fitted parameters.

For every mouse and every generating model G in {M1, M2, M3}: take that mouse's robust-variant fit of G (lapse
mixture, inferred prior, wide bounds; results/c1/c1_robust_best_fits.csv), replay the mouse's own trials through it
once (same trial count, same inferred-prior covariate, lapses included), then fit M1, M2 and M3 to the synthetic data
with exactly the robust settings and re-score them on a common seed, as in the real analysis.

This asks the question the C1 claims depend on: if a mouse really were M1 (or M2, M3) with the parameters we fitted,
would the pipeline say so? It is conservative: each synthetic data set gets one fit seed (the real fits got three).

    python scripts/validate_robust_recovery.py --n-jobs 38            # 10 mice x 3 generators x 3 fits = 90 fits
    python scripts/validate_robust_recovery.py --smoke                # seconds, end-to-end check

Writes results/recovery/robust_model_recovery.csv (one row per fit; resumable) and
results/recovery/robust_model_recovery_summary.md (confusion matrix, delta-BIC under each generator).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run_c1 import RESCORE, SMOKE_FIT, SMOKE_RESCORE, _md_table, load_cohort, variant_overrides  # noqa: E402
from src.optimization.empirical import EMPIRICAL_FIT_DEFAULTS, fit_mouse, fitted_params, simulate_from_fit  # noqa: E402
from src.optimization.fit import score_ddm  # noqa: E402

MODELS = ("M1", "M2", "M3")
KEY = ["subject_id", "true_model", "fit_model"]


def synthetic(trials: pd.DataFrame, params: dict, model: str, seed: int) -> pd.DataFrame:
    """One synthetic data set of the mouse's size, with the model's (inferred) prior as the fitting covariate."""
    sim = simulate_from_fit(trials, params, model, "fixed", 2.5, n_rep=1, seed=seed, prior_col="prior_model")
    return sim.assign(prior=sim["prior_model"], subject_id=str(trials["subject_id"].iloc[0]))


def one_fit(subject: str, trials: pd.DataFrame, true_model: str, params: dict, fit_model: str, gen_seed: int,
            fit_seed: int, overrides: dict, rescore: dict) -> dict:
    data = synthetic(trials, params, true_model, gen_seed)
    res = fit_mouse(data, fit_model, "fixed", 2.5, seed=fit_seed, **overrides)
    cfg = {k: v for k, v in {**EMPIRICAL_FIT_DEFAULTS, **overrides}.items() if k not in ("n_sim_per_cond", "n_sim_final", "seed")}
    sc = score_ddm(data, res["params"], model_type=fit_model, bound_type="fixed", t_max=2.5,
                   n_sim_per_cond=rescore["n_sim_per_cond"], seed=rescore["seed"], **cfg)
    row = {"subject_id": subject, "true_model": true_model, "fit_model": fit_model, "n_trials": len(data),
           "gen_seed": gen_seed, "fit_seed": fit_seed, "bic_fit": res["bic"], "bic": sc["bic"], "nll": sc["nll"]}
    row.update({f"p_{k}": v for k, v in res["params"].items()})
    return row


def summarize(df: pd.DataFrame, observed: pd.DataFrame = None) -> str:
    wide = df.pivot_table(index=["subject_id", "true_model"], columns="fit_model", values="bic").dropna()
    wide["winner"] = wide[list(MODELS)].idxmin(axis=1)
    conf = pd.crosstab(wide.index.get_level_values("true_model"), wide["winner"]).reindex(index=MODELS, columns=MODELS, fill_value=0)
    conf.index.name, conf.columns.name = "true", "winner"
    rate = (np.diag(conf.values) / conf.sum(axis=1).clip(lower=1).values)
    wide["d_M1_M2"] = wide["M1"] - wide["M2"]
    wide["d_M3_M2"] = wide["M3"] - wide["M2"]
    stats = wide.groupby(level="true_model")[["d_M1_M2", "d_M3_M2"]].agg(["mean", "median", "min", "max"]).round(1)
    lines = ["# Model recovery for the robust C1 model class (fitted parameters, each mouse's own design)", "",
             f"{len(wide)} synthetic data sets ({wide.index.get_level_values('subject_id').nunique()} mice x generators), "
             "each fitted with M1, M2, M3 (robust settings, one fit seed, common-seed re-scoring).", "",
             "Confusion matrix (rows = generating model, columns = lowest BIC):", "", _md_table(conf.reset_index()), "",
             "P(recovered | true): " + ", ".join(f"{m} {r:.0%}" for m, r in zip(MODELS, rate)), "",
             "Delta BIC under each generating model (positive d_M1_M2 = M1 worse than M2):", "", stats.to_string(), ""]
    if observed is not None:
        lines += ["Observed (real mice, robust variant): mean dBIC M1-M2 = "
                  f"{observed['dbic_M1_vs_M2'].mean():.1f}, M3-M2 = {observed['dbic_M3_vs_M2'].mean():.1f}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--cohort", default=os.path.join(ROOT, "data", "processed", "cohort_rt2p5.parquet"))
    ap.add_argument("--extended", default=os.path.join(ROOT, "data", "processed", "cohort_extended.parquet"))
    ap.add_argument("--fits", default=os.path.join(ROOT, "results", "c1", "c1_robust_best_fits.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recovery", "robust_model_recovery.csv"))
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--smoke", action="store_true", help="2 mice, tiny search, written next to --out with a _smoke suffix")
    a = ap.parse_args()

    cohort = load_cohort(a.cohort, inferred=True, extended=a.extended)
    best = pd.read_csv(a.fits)
    best = best[(best["bound"] == "fixed") & (best["model"].isin(MODELS))]
    overrides = variant_overrides("robust")
    rescore = dict(RESCORE)
    subjects = sorted(best["subject_id"].unique())
    out = a.out
    if a.smoke:
        overrides = {**overrides, **SMOKE_FIT}
        rescore = dict(SMOKE_RESCORE)
        subjects = subjects[:2]
        out = out.replace(".csv", "_smoke.csv")
        cohort = cohort.groupby("subject_id", group_keys=False).head(1500)

    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=KEY)
    have = set(map(tuple, done[KEY].values.tolist()))
    rng = np.random.default_rng(a.seed)
    tasks = []
    for s in subjects:
        trials = cohort[cohort["subject_id"] == s].reset_index(drop=True)
        for g in MODELS:
            gen_seed = int(rng.integers(1, 2**31 - 1))
            row = best[(best["subject_id"] == s) & (best["model"] == g)]
            if row.empty:
                continue
            params = fitted_params(row.iloc[0])
            for f in MODELS:
                fit_seed = int(rng.integers(1, 2**31 - 1))
                if (s, g, f) not in have:
                    tasks.append((s, trials, g, params, f, gen_seed, fit_seed, overrides, rescore))
    tasks.sort(key=lambda t: -(3.0 if t[4] == "M3" else 1.0))      # slow M3 fits first
    print(f"{len(tasks)} fits to run ({len(have)} already done) -> {out}", flush=True)

    rows = done.to_dict("records")
    if tasks:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for row in Parallel(n_jobs=a.n_jobs, verbose=5, return_as="generator_unordered")(delayed(one_fit)(*t) for t in tasks):
            rows.append(row)
            tmp = out + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out)
    df = pd.DataFrame(rows)
    obs_path = os.path.join(os.path.dirname(a.fits), "c1_robust_bic_table.csv")
    observed = pd.read_csv(obs_path) if os.path.exists(obs_path) else None
    text = summarize(df, observed)
    with open(out.replace(".csv", "_summary.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
