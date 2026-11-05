"""
Robustness check for C1: is M1's near-zero fitted start-point bias z an optimizer artefact?
Refits M1 with z FIXED at 0.15 / 0.30 / 0.45 (all other parameters free, same heavy search) for
selected mice and compares the final NLL with the free M1 and M2 fits in results/c1/c1_best_fits.csv.
Writes results/c1/c1_m1_fixed_z_check.csv. ~10 min with 6 workers (heavy search, one fit per job).

    python scripts/check_m1_start_point.py [--mice NYU-09 ibl_witten_13] [--z 0.15 0.3 0.45]
"""
import argparse
import os

import pandas as pd
from joblib import Parallel, delayed

from src.optimization.empirical import fit_mouse


def _one(df, subject, z):
    r = fit_mouse(df, "M1", "fixed", 2.5, fixed_params={"z": float(z)})
    return {"subject_id": subject, "z_fixed": z, "nll": r["nll"], "bic_free_params": r["bic"], "n_params": r["n_params"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mice", nargs="+", default=["NYU-09", "ibl_witten_13"])
    p.add_argument("--z", nargs="+", type=float, default=[0.15, 0.30, 0.45])
    p.add_argument("--cohort", default="data/processed/cohort_rt2p5.parquet")
    p.add_argument("--best", default="results/c1/c1_best_fits.csv")
    p.add_argument("--out", default="results/c1/c1_m1_fixed_z_check.csv")
    p.add_argument("--n-jobs", type=int, default=6)
    a = p.parse_args()
    cohort = pd.read_parquet(a.cohort)
    jobs = [(cohort[cohort.subject_id == s].reset_index(drop=True), s, z) for s in a.mice for z in a.z]
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_one)(*j) for j in jobs)
    out = pd.DataFrame(rows)
    best = pd.read_csv(a.best)
    for m in ("M1", "M2"):
        ref = best[best.model == m].set_index("subject_id")["nll"]
        out[f"nll_{m}_free"] = out["subject_id"].map(ref)
    out["dnll_vs_free_M1"] = out["nll"] - out["nll_M1_free"]
    out["dnll_vs_M2"] = out["nll"] - out["nll_M2_free"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(1).to_string())


if __name__ == "__main__":
    main()
