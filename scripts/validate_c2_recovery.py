"""
Bound-type recovery for C2: can the fitting pipeline tell a FIXED from a Weibull-COLLAPSING bound, and does a heavy RT tail
make it hallucinate urgency?

Datasets are simulated from M2 with the parameter ranges the C1 fits of the real mice produced, replaying the trial design
(contrast, block prior, previous choice) of a real mouse, and are then RT-filtered exactly like the real data
(0.1 s <= RT <= cutoff). Three generating processes:
    fixed              fixed bound
    weibull            Weibull collapsing bound (a(t)/a = 0.85 at 1 s, 0.37 at 2.5 s)
    fixed+slow_lapses  fixed bound, but 8% of the trials are "disengaged": RT = 1 s + Exp(1.5 s), random choice
                       (real mice have a heavy RT tail that a diffusion model cannot produce)
Each dataset is fitted with the SAME pipeline as the real data (`fit_cohort`: fixed fit, then Weibull fit warm-started from
it) under three configurations: the C1 likelihood at 2.5 s, the truncation-aware likelihood at 2.5 s and at 5.0 s.

    python scripts/validate_c2_recovery.py --n-datasets 4 --n-jobs 12     # ~1.6 h on 12 workers; resumable

Writes results/c2/validation/c2_bound_recovery.csv (one row per dataset x configuration) and
c2_bound_recovery_summary.csv / .md (false-positive rate, power, fitted collapse).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.simulator import DDMSimulator  # noqa: E402
from src.optimization import c1, c2  # noqa: E402
from src.optimization.empirical import COLLAPSE_STARTS, fit_cohort  # noqa: E402

BASE = {"a": 0.9, "t0": 0.12, "k": 3.0, "beta": 0.35, "v_h": 0.4, "v_0": 0.1, "v_b": 0.4, "s_v": 1.5, "s_z": 0.4}
TRUTHS = {
    "fixed": dict(params=BASE, bound="fixed", contam=0.0),
    "weibull": dict(params={**BASE, "a": 1.1, "lam": 2.5, "gamma": 2.0}, bound="weibull", contam=0.0),
    "fixed+slow_lapses": dict(params=BASE, bound="fixed", contam=0.08),
}
CONFIGS = {   # name: (rt_max, truncation-aware likelihood)
    "2.5s_c1_likelihood": (2.5, False),
    "2.5s_truncated": (2.5, True),
    "5.0s_truncated": (5.0, True),
}
SMOKE_FIT = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)


def make_datasets(design: pd.DataFrame, n_datasets: int, seed: int) -> pd.DataFrame:
    """All simulated datasets stacked in one §3.1 table (subject_id = '<truth>#<i>'), filtered to 0.1 <= RT <= 5.0 s."""
    rng = np.random.default_rng(seed)
    parts = []
    for truth, spec in TRUTHS.items():
        for i in range(n_datasets):
            sim = DDMSimulator(dt=0.002, t_max=8.0).simulate(
                spec["params"], design["contrast"].to_numpy(float), design["prior"].to_numpy(float),
                design["prev_choice"].to_numpy(float), model_type="M2", bound_type=spec["bound"], rng=rng)
            if spec["contam"] > 0:
                lapse = rng.random(len(sim)) < spec["contam"]
                sim.loc[lapse, "rt"] = 1.0 + rng.exponential(1.5, lapse.sum())
                sim.loc[lapse, "choice"] = rng.choice([-1, 1], lapse.sum()).astype(np.int8)
            sim = sim[(sim["rt"] >= 0.1) & (sim["rt"] <= 5.0)].reset_index(drop=True)
            sim["trial_idx"] = np.arange(len(sim), dtype=np.int32)
            sim["subject_id"] = f"{truth}#{i}"
            parts.append(sim)
    return pd.concat(parts, ignore_index=True)


def run(args) -> pd.DataFrame:
    real = pd.read_parquet(os.path.join(ROOT, "data", "processed", "cohort_rt5p0.parquet"))
    design = real[real["subject_id"] == args.mouse].head(args.n_trials).reset_index(drop=True)
    data = make_datasets(design, args.n_datasets, args.seed)
    os.makedirs(args.out, exist_ok=True)
    extra = dict(SMOKE_FIT) if args.smoke else {}
    rows = []
    for cfg, (rt_max, truncate) in CONFIGS.items():
        cohort = data[data["rt"] <= rt_max]
        cache = os.path.join(args.out, f"fits_{cfg}.csv")
        kw = dict(models=("M2",), rt_max=rt_max, n_jobs=args.n_jobs, cache_path=cache, seeds=[args.seed], **extra)
        if truncate:
            kw["truncate"] = True
        fixed = fit_cohort(cohort, bounds=("fixed",), **kw)
        fits = fit_cohort(cohort, bounds=("weibull",), init_from=fixed[fixed["bound"] == "fixed"], init_extra=COLLAPSE_STARTS, **kw)
        tab = c2.bound_table(c1.collapse_replicates(fits), "M2", rt_max)
        rows.append(tab.assign(config=cfg, truth=tab["subject_id"].str.split("#").str[0]))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(args.out, "c2_bound_recovery.csv"), index=False)
    return out


def summarise(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cfg, truth), g in out.groupby(["config", "truth"], sort=False):
        rows.append({
            "config": cfg, "truth": truth, "n_datasets": len(g),
            "n_weibull_wins": int((g["winner"] == "weibull").sum()),
            "n_weibull_wins_by_10": int((g["dbic_weibull_vs_fixed"] < -10).sum()),
            "mean_dbic": g["dbic_weibull_vs_fixed"].mean(), "median_lam": g["p_lam"].median(),
            "median_frac_t1": g["frac_t1"].median(), "median_frac_t2.5": g["frac_t2.5"].median(),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-datasets", type=int, default=4, help="datasets per generating process")
    ap.add_argument("--n-trials", type=int, default=12000, help="trials of the replayed design before filtering")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--mouse", default="NYU-09", help="real mouse whose trial covariates are replayed")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "c2", "validation"))
    ap.add_argument("--smoke", action="store_true", help="tiny fits (seconds) to test the script")
    a = ap.parse_args(argv)
    if a.smoke:
        a.n_datasets, a.n_trials = 1, 2000
    out = run(a)
    summ = summarise(out)
    summ.to_csv(os.path.join(a.out, "c2_bound_recovery_summary.csv"), index=False)
    from run_c2 import _md_table
    with open(os.path.join(a.out, "c2_bound_recovery_summary.md"), "w") as f:
        f.write("# C2 bound-type recovery (M2; truth known)\n\n" + _md_table(summ) + "\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
