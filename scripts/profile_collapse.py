"""
Profile likelihood of the collapse latency lam (C2 diagnostic).

If a Weibull fit ends at 'no collapse' (lam at its ceiling) it is natural to ask whether the data really prefer a fixed bound or
whether Nelder-Mead simply missed a collapsing optimum. Here lam is FIXED at each value of a grid (gamma and all other parameters
free, warm-started from the same mouse's fixed-bound fit) and the model is refitted; the resulting NLL profile over lam, with the
fixed-bound NLL as reference, shows how much worse every degree of collapse is (or how much better).

    python scripts/profile_collapse.py --mice NYU-09 KS019 --rt-max 5.0 --n-jobs 12     # ~20 min

Writes results/c2/c2_collapse_profile_<rt_max>.csv (columns: subject_id, lam, nll, bic, gamma, a, dnll_vs_fixed).
"""
import argparse
import os
import sys

import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.optimization.empirical import fit_mouse  # noqa: E402

SMOKE_FIT = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)


def _fit(subject, df, rt_max, lam, init, truncate, extra):
    kw = dict(seed=12345, truncate=truncate, **extra)
    if lam is None:
        r = fit_mouse(df, "M2", "fixed", rt_max, **kw)
    else:
        guesses = [{**init, "lam": lam, "gamma": g} for g in (1.5, 2.5, 3.5)]
        r = fit_mouse(df, "M2", "weibull", rt_max, fixed_params={"lam": lam}, init_params=guesses, **kw)
    return {"subject_id": subject, "lam": lam if lam is not None else float("inf"), "nll": r["nll"], "bic": r["bic"],
            "gamma": r["params"].get("gamma", float("nan")), "a": r["params"]["a"], "params": r["params"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mice", nargs="+", default=["NYU-09", "KS019"])
    ap.add_argument("--rt-max", type=float, default=5.0)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    ap.add_argument("--no-truncate", action="store_true", help="use the C1 likelihood instead of the truncation-aware one")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    name = "cohort_rt5p0" if a.rt_max > 2.5 else "cohort_rt2p5"
    cohort = pd.read_parquet(os.path.join(ROOT, "data", "processed", f"{name}.parquet"))
    extra = dict(SMOKE_FIT) if a.smoke else {}
    truncate = not a.no_truncate
    data = {m: cohort[cohort["subject_id"] == m].reset_index(drop=True) for m in a.mice}

    fixed = Parallel(n_jobs=min(a.n_jobs, len(a.mice)))(delayed(_fit)(m, data[m], a.rt_max, None, {}, truncate, extra) for m in a.mice)
    ref = {r["subject_id"]: r for r in fixed}
    tasks = [(m, data[m], a.rt_max, lam, ref[m]["params"], truncate, extra) for m in a.mice for lam in a.lams]
    prof = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_fit)(*t) for t in tasks)
    rows = pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in fixed + prof])
    rows["dnll_vs_fixed"] = rows["nll"] - rows["subject_id"].map({m: ref[m]["nll"] for m in ref})
    out = a.out or os.path.join(ROOT, "results", "c2", f"c2_collapse_profile_{a.rt_max:g}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows.sort_values(["subject_id", "lam"]).to_csv(out, index=False)
    print(rows.sort_values(["subject_id", "lam"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
