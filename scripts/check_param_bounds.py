"""
Robustness check for C1: do the search bounds (t0 >= 0.08 s, s_z <= 0.6) limit the fits?
Refits M2 (same heavy search, same two seeds as the main run) with t0 in [0.02, 0.6] and s_z in [0, 1.2] for mice
whose main fits sat on those bounds and compares BIC/NLL with the main fits in results/c1/c1_fits.csv.
Writes results/c1/c1_bounds_check.csv.   python scripts/check_param_bounds.py [--mice CSHL045 ZM_1743] [--models M2]
"""
import argparse
import os

import pandas as pd
from joblib import Parallel, delayed

from src.optimization.empirical import WIDE_BOUNDS as WIDE, fit_mouse


def _one(df, subject, model, seed):
    r = fit_mouse(df, model, "fixed", 2.5, seed=seed, param_bounds=WIDE)
    return {"subject_id": subject, "model": model, "seed": seed, "nll_wide": r["nll"], "bic_wide": r["bic"],
            **{f"wide_{k}": v for k, v in r["params"].items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mice", nargs="+", default=["CSHL045", "ZM_1743"])
    p.add_argument("--models", nargs="+", default=["M2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[12345, 777])
    p.add_argument("--cohort", default="data/processed/cohort_rt2p5.parquet")
    p.add_argument("--fits", default="results/c1/c1_fits.csv")
    p.add_argument("--out", default="results/c1/c1_bounds_check.csv")
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()
    cohort = pd.read_parquet(a.cohort)
    jobs = [(cohort[cohort.subject_id == s].reset_index(drop=True), s, m, sd) for s in a.mice for m in a.models for sd in a.seeds]
    out = pd.DataFrame(Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_one)(*j) for j in jobs))
    main_fits = pd.read_csv(a.fits)
    main_fits = main_fits[["subject_id", "model", "seed", "nll", "bic"] + [c for c in main_fits if c.startswith("p_")]]
    out = out.merge(main_fits.rename(columns={"nll": "nll_default", "bic": "bic_default"}), on=["subject_id", "model", "seed"], how="left")
    out["dbic_wide_minus_default"] = out["bic_wide"] - out["bic_default"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out[["subject_id", "model", "seed", "bic_default", "bic_wide", "dbic_wide_minus_default", "wide_t0", "p_t0", "wide_s_z", "p_s_z"]].round(2).to_string())


if __name__ == "__main__":
    main()
