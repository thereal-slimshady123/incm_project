"""
Model recovery for the Week 3 EMPIRICAL model class (contrast exponent, trial variabilities and a
neutral side bias all free) on a realistic design: the trial covariates (contrast, block prior,
previous choice) are replayed from a real mouse. Parth's `run_model_recovery` validates the
plain 5-parameter models at N = 1000; this checks that M1 / M2 / M3 stay distinguishable once the
nuisance parameters that real data need are free, at realistic trial counts.

    python scripts/validate_empirical_recovery.py --n-datasets 6 --n-trials 6000 --n-jobs 18

Writes results/recovery/empirical_model_recovery.csv and prints the confusion matrix.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simulator import DDMSimulator  # noqa: E402
from src.optimization.empirical import fit_mouse  # noqa: E402

MODELS = ("M1", "M2", "M3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_truth(model: str, rng: np.random.Generator) -> dict:
    """Parameters in the range the pilot fits of real mice produced, with a visible prior effect."""
    p = {
        "a": rng.uniform(0.9, 1.3), "t0": rng.uniform(0.12, 0.2), "k": rng.uniform(2.5, 4.5),
        "beta": rng.uniform(0.15, 0.45), "v_h": rng.uniform(0.0, 0.5), "v_0": rng.uniform(-0.2, 0.2),
        "s_v": rng.uniform(0.8, 2.0), "s_z": rng.uniform(0.05, 0.35),
    }
    if model == "M1":
        p["z"] = rng.uniform(0.15, 0.35)
    elif model == "M2":
        p["v_b"] = rng.uniform(0.5, 1.0)
    else:
        p["b_max"] = rng.uniform(0.8, 1.6)
        p["tau_b"] = float(rng.choice([0.15, 0.35, 0.8]))
    return {k: float(v) for k, v in p.items()}


def one_dataset(idx: int, true_model: str, template: pd.DataFrame, seed: int, fit_kwargs: dict) -> dict:
    rng = np.random.default_rng(seed)
    truth = sample_truth(true_model, rng)
    sim = DDMSimulator(dt=0.002, t_max=2.5).simulate(
        truth, template["contrast"].to_numpy(float), template["prior"].to_numpy(float),
        template["prev_choice"].to_numpy(float), model_type=true_model, rng=rng,
    )
    sim = sim[sim["rt"] <= 2.5].reset_index(drop=True)
    sim["trial_idx"] = np.arange(len(sim), dtype=np.int32)
    c0 = sim[sim["contrast"] == 0]
    prior_effect = (c0[c0["prior"] > 0]["choice"] == 1).mean() - (c0[c0["prior"] < 0]["choice"] == 1).mean()
    rec = {"idx": idx, "true_model": true_model, "prior_effect_c0": prior_effect, "seed": seed}
    rec.update({f"true_{k}": v for k, v in truth.items()})
    for m in MODELS:
        r = fit_mouse(sim, m, "fixed", 2.5, seed=seed + 1, **fit_kwargs)
        rec[f"bic_{m}"] = r["bic"]
    rec["winner"] = min(MODELS, key=lambda m: rec[f"bic_{m}"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-datasets", type=int, default=6, help="datasets per generating model")
    ap.add_argument("--n-trials", type=int, default=6000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--mouse", default="NYU-09", help="real mouse whose trial covariates are replayed")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recovery", "empirical_model_recovery.csv"))
    a = ap.parse_args()

    real = pd.read_parquet(os.path.join(ROOT, "data", "processed", "cohort_rt2p5.parquet"))
    template = real[real["subject_id"] == a.mouse].head(a.n_trials).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    tasks = [(i, m, template, int(rng.integers(1, 10**6)), {}) for m in MODELS for i in range(a.n_datasets)]
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_dataset)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)

    conf = pd.crosstab(df["true_model"], df["winner"]).reindex(index=MODELS, columns=MODELS, fill_value=0)
    print("\nConfusion matrix (rows = generating model, columns = winning model by BIC):\n", conf)
    print("\nP(win = true | true):", (np.diag(conf.values) / conf.sum(axis=1).values).round(2))
    print("Mean prior effect at 0% contrast of the simulated data:", df.groupby("true_model")["prior_effect_c0"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
