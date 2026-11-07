"""
Command-line runner for Week 3, C2 (urgency / boundary dynamics), Track B: fit fixed vs Weibull-collapsing bounds
to the 10 IBL mice at the 2.5 s and 5.0 s RT cutoffs, test whether the C1 conclusion survives collapsing bounds,
and analyse the RT tail.
Author: Joshua Koilpillai (2024101128)

Usage:
    python run_c2.py fit                          # main fits, all default variants (hours; resumable, see table below)
    python run_c2.py fit --variant trunc_5p0      # one variant
    python run_c2.py fit --variant trunc_2p5 --models M1 M2 M3   # add the invariance models under the C2 likelihood
    python run_c2.py analyze                      # tables, tests, posterior-predictive CBF and tails, figures (~2 min)
    python run_c2.py all
    python run_c2.py smoke                        # tiny end-to-end check on 3 mice (seconds)

Variants (each one is a cache file results/c2/c2_fits_<variant>.csv holding the fixed AND the Weibull fits):
    trunc_2p5   2.5 s cutoff, truncation-aware likelihood, M2          <- headline fixed-vs-collapsing comparison
    trunc_5p0   5.0 s cutoff, truncation-aware likelihood, M2          <- headline, extended cutoff + tail analysis
    c1like_2p5  2.5 s cutoff, the C1 likelihood, M1/M2/M3              <- invariance test; its fixed fits are COPIED from the C1 fits
    tail_5p0    5.0 s cutoff, truncation-aware, extra QMP quantiles .95/.99, M2   <- sensitivity: tail-weighted likelihood (opt-in)
    trunc_2p5_wide, trunc_5p0_wide   as trunc_*, with the relaxed t0 / s_z search bounds of the C1 `wide_bounds` check (opt-in)

Every Weibull fit is warm-started from the same mouse/model/seed's fixed fit (see src/optimization/empirical.py).
Then a symmetric POLISH pass (disable with --no-polish) re-optimises the fixed model from the Weibull solution and the Weibull
model from that polished fixed solution, because the bound-recovery check (scripts/validate_c2_recovery.py) showed that either
fit can end in a worse local optimum (+-50 NLL) and the nested Weibull then looks spuriously worse or better than the fixed bound.
Polish replicates are stored with seed + 1,000,000; the best-BIC replicate of each (mouse, model, bound) is what the analysis uses.
"""

import argparse
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from src.optimization.empirical import COLLAPSE_STARTS, WIDE_BOUNDS, fit_cohort

OUT_DIR = "results/c2"
C1_FITS = "results/c1/c1_fits.csv"
COHORT_2P5 = "data/processed/cohort_rt2p5.parquet"
COHORT_5P0 = "data/processed/cohort_rt5p0.parquet"
DEFAULT_SEEDS = (12345, 777)
TAIL_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
POLISH_SEED_OFFSET = 1_000_000      # polish replicate of seed s is stored as seed s + this offset
POLISH_FIT = dict(n_screen=0, n_starts=1, n_restarts=2)     # local refinement only, from the given starting points
SMOKE_FIT = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)

VARIANTS: Dict[str, dict] = {
    "trunc_2p5": dict(cohort=COHORT_2P5, rt_max=2.5, truncate=True, models=("M2",)),
    "trunc_5p0": dict(cohort=COHORT_5P0, rt_max=5.0, truncate=True, models=("M2",)),
    "c1like_2p5": dict(cohort=COHORT_2P5, rt_max=2.5, truncate=False, models=("M1", "M2", "M3"), reference=C1_FITS),
    "tail_5p0": dict(cohort=COHORT_5P0, rt_max=5.0, truncate=True, models=("M2",), quantiles=TAIL_QUANTILES),
    "trunc_2p5_wide": dict(cohort=COHORT_2P5, rt_max=2.5, truncate=True, models=("M2",), param_bounds=WIDE_BOUNDS),
    "trunc_5p0_wide": dict(cohort=COHORT_5P0, rt_max=5.0, truncate=True, models=("M2",), param_bounds=WIDE_BOUNDS),
}
LABELS = {"trunc_2p5": "2.5 s cutoff, truncation-aware", "trunc_5p0": "5.0 s cutoff, truncation-aware",
          "c1like_2p5": "2.5 s cutoff, C1 likelihood", "tail_5p0": "5.0 s cutoff, tail-weighted quantiles",
          "trunc_2p5_wide": "2.5 s, truncation-aware, relaxed bounds", "trunc_5p0_wide": "5.0 s, truncation-aware, relaxed bounds"}
DEFAULT_VARIANTS = ("trunc_2p5", "trunc_5p0", "c1like_2p5")


def cache_path(out: str, variant: str) -> str:
    return os.path.join(out, f"c2_fits_{variant}.csv")


def seed_from_reference(cache: str, reference: str, rt_max: float) -> None:
    """Start a variant's cache with the fixed-bound fits of another run made with IDENTICAL settings (the C1 fits)."""
    if os.path.exists(cache) or not os.path.exists(reference):
        return
    ref = pd.read_csv(reference)
    ref = ref[(ref["bound"] == "fixed") & (ref["rt_max"] == rt_max)].assign(truncated=False)
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    ref.to_csv(cache, index=False)
    print(f"seeded {os.path.relpath(cache)} with {len(ref)} fixed-bound fits from {reference}")


def run_fit_variant(name: str, args) -> pd.DataFrame:
    spec = VARIANTS[name]
    cohort = pd.read_parquet(spec["cohort"])
    cache = cache_path(args.out, name)
    models = tuple(args.models) if args.models else spec["models"]
    kwargs = dict(SMOKE_FIT) if args.smoke else {}
    if spec.get("truncate"):
        kwargs["truncate"] = True
    if "quantiles" in spec:
        kwargs["quantiles"] = spec["quantiles"]
    if "param_bounds" in spec:
        kwargs["param_bounds"] = spec["param_bounds"]
    subjects = sorted(cohort["subject_id"].unique())[:3] if args.smoke else None
    if "reference" in spec and not args.smoke:
        seed_from_reference(cache, spec["reference"], spec["rt_max"])
    common = dict(models=models, rt_max=spec["rt_max"], n_jobs=args.n_jobs, cache_path=cache, subjects=subjects, **kwargs)
    t = time.time()
    print(f"[{name}] phase 1/4: fixed bound, models {models}")
    fixed = fit_cohort(cohort, bounds=("fixed",), seeds=args.seeds, **common)
    print(f"[{name}] phase 2/4: Weibull bound, warm-started from the fixed fits")
    fits = fit_cohort(cohort, bounds=("weibull",), seeds=args.seeds, init_from=fixed[fixed["bound"] == "fixed"],
                      init_extra=COLLAPSE_STARTS, **common)
    if not args.no_polish:
        pseeds = [int(s) + POLISH_SEED_OFFSET for s in args.seeds]
        pkw = {**common, **POLISH_FIT}
        shift = lambda df: df.assign(seed=df["seed"].astype(int) + POLISH_SEED_OFFSET)
        print(f"[{name}] phase 3/4: polish the fixed fit from the Weibull solution")
        fits = fit_cohort(cohort, bounds=("fixed",), seeds=pseeds, init_from=shift(fits[fits["bound"] == "weibull"]), **pkw)
        print(f"[{name}] phase 4/4: polish the Weibull fit from the polished fixed solution")
        fixed_p = fits[(fits["bound"] == "fixed") & (fits["seed"].isin(pseeds))]
        fits = fit_cohort(cohort, bounds=("weibull",), seeds=pseeds, init_from=fixed_p, init_extra=COLLAPSE_STARTS, **{**pkw, "n_starts": 3})
    print(f"[{name}] {len(fits)} rows in {os.path.relpath(cache)} ({(time.time() - t) / 60:.1f} min this call)")
    return fits


def _md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run_analyze(args) -> None:
    """All C2 tables, tests, posterior-predictive checks and figures from the variants' fit caches."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 0       # ~40 figures are written in one go
    from src.optimization import c1, c2
    from src.visualization.plots_c1 import plot_cbf_overlay_cohort
    from src.visualization.plots_c2 import (
        plot_bound_comparison, plot_fitted_bounds, plot_invariance, plot_tail_by_mouse, plot_tail_diagnostics,
    )

    out = args.out
    fig_dir = args.fig_dir or (os.path.join(out, "figures") if args.smoke else os.path.join("results", "figures", "c2"))
    os.makedirs(fig_dir, exist_ok=True)

    def save(df, name):
        df.to_csv(os.path.join(out, name), index=False)

    lines = ["# C2 results: fixed vs Weibull-collapsing bounds, and invariance of the C1 conclusion", ""]
    bound_tabs: Dict[str, Dict[str, pd.DataFrame]] = {}      # variant -> model -> bound_table
    thresholds: Dict[str, float] = {}
    cohorts: Dict[float, pd.DataFrame] = {}
    cbf_verdicts: Dict[str, Dict[str, dict]] = {}

    for name, spec in VARIANTS.items():
        path = cache_path(out, name)
        if not os.path.exists(path):
            continue
        fits = pd.read_csv(path)
        independent = fits[fits["seed"] < POLISH_SEED_OFFSET]       # polish replicates start from other fits: not independent runs
        rt_max = spec["rt_max"]
        cohort = cohorts.setdefault(rt_max, pd.read_parquet(spec["cohort"]))
        best = c1.collapse_replicates(fits)
        noise = c1.noise_floor(independent)
        thr = c1.decisive_threshold(noise["sd"])
        thresholds[name] = thr
        lik = "truncation-aware" if spec.get("truncate") else "C1 (untruncated)"
        pre = f"c2_{name}"
        save(best, f"{pre}_best_fits.csv")
        save(pd.DataFrame([{**noise, "decisive_threshold": thr}]), f"{pre}_noise_floor.csv")
        save(pd.concat([c1.parameter_table(best, rt_max, b).assign(bound=b) for b in c2.BOUNDS if (best["bound"] == b).any()]), f"{pre}_parameters.csv")
        lines += [f"## Variant `{name}`: cutoff {rt_max} s, {lik} likelihood", "",
                  f"{int(fits.groupby(c1.KEYS).ngroups)} (mouse, model, bound) fits x up to {int(best['n_reps'].max())} replicates "
                  f"(independent seeds plus polish passes); best-BIC replicate kept. "
                  f"Run-to-run SD of BIC between the independent seeds = {noise['sd']:.1f}; decisive if |delta BIC| > {thr:.1f}.", ""]

        # --- fixed vs Weibull for every model with both bounds ------------------------------------
        bound_tabs[name] = {}
        all_tests = []
        for model in c1.MODELS:
            have = set(best[(best["model"] == model) & (best["rt_max"] == rt_max)]["bound"])
            if not set(c2.BOUNDS) <= have:
                continue
            tab = c2.bound_table(best, model, rt_max, threshold=thr)
            bound_tabs[name][model] = tab
            tests = c2.bound_tests(tab).assign(model=model)
            all_tests.append(tests)
            save(tab, f"{pre}_bound_table_{model}.csv")
            lines += [f"### {model}: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)", "",
                      _md_table(tab[["subject_id", "bic_fixed", "bic_weibull", "dbic_weibull_vs_fixed", "dnll_weibull_vs_fixed", "winner", "evidence",
                                     "p_lam", "p_gamma", "frac_t1", "frac_t2.5", "lam_at_ceiling"]], ".6g"), "",
                      _md_table(tests.drop(columns=["sum_delta"], errors="ignore")), ""]
        if all_tests:
            save(pd.concat(all_tests, ignore_index=True), f"{pre}_bound_tests.csv")

        # --- invariance of the C1 comparison under the Weibull bound -------------------------------
        full = all(m in bound_tabs[name] for m in c1.MODELS)
        if full:
            inv = c2.invariance_table(best, rt_max, threshold=thr)
            inv_tests = c2.invariance_tests(best, inv, rt_max)
            corr = c2.invariance_correlations(inv)
            emp = c1.empirical_cbf(cohort)
            verdicts, cbf_tabs = {}, {}
            for b in c2.BOUNDS:
                pred = c1.predicted_cbf(cohort, best, bound=b, rt_max=rt_max, n_rep=args.n_rep, emp=emp)
                fit_tab = c1.cbf_fit_table(emp, pred)
                cbf_tabs[b] = (pred, fit_tab)
                save(pred, f"{pre}_cbf_predicted_{b}.csv")
                save(fit_tab.assign(bound=b), f"{pre}_cbf_fit_{b}.csv")
                tests_b = inv_tests[inv_tests["bound"] == b]
                verdicts[b] = c2.c1_verdict(tests_b, fit_tab)
                plot_cbf_overlay_cohort(emp, pred, title=f"C2 ({name}): model-predicted CBF, {c2.BOUND_LABELS[b].lower()}",
                                        save_path=os.path.join(fig_dir, f"c2_{name}_cbf_overlay_{b}.png"))
            cbf_verdicts[name] = verdicts
            save(inv, f"{pre}_invariance_table.csv")
            save(inv_tests, f"{pre}_invariance_tests.csv")
            save(corr, f"{pre}_invariance_correlations.csv")
            save(pd.DataFrame(verdicts).T.reset_index().rename(columns={"index": "bound"}), f"{pre}_c1_verdict.csv")
            plot_invariance(inv, title=f"C2 ({name}): C1 delta-BIC under fixed vs Weibull bound",
                            save_path=os.path.join(fig_dir, f"c2_{name}_invariance.png"))
            same = verdicts["fixed"] == verdicts["weibull"]
            lines += ["### Invariance of the C1 comparison (M1 / M2 / M3) under the Weibull bound", "",
                      _md_table(inv[["subject_id", "dbic_M1_vs_M2_fixed", "dbic_M1_vs_M2_weibull", "dbic_M3_vs_M2_fixed", "dbic_M3_vs_M2_weibull",
                                     "winner_fixed", "winner_weibull", "same_winner"]], ".6g"), "",
                      "Across-mouse tests under each bound, and tests of the shift between bounds (bound = shift):", "",
                      _md_table(inv_tests.drop(columns=["sum_delta"], errors="ignore")), "",
                      "Rank agreement of the per-mouse deltas between bounds:", "", _md_table(corr), "",
                      f"C1 verdict under the fixed bound: {verdicts['fixed']}", "",
                      f"C1 verdict under the Weibull bound: {verdicts['weibull']}", "",
                      f"**Invariant: {same}**", ""]

        # --- RT tail (extended cutoff only: needs data beyond 2.5 s) --------------------------------------------
        if rt_max >= 4.5 and "M2" in bound_tabs[name]:
            tt = c2.tail_tables(cohort, best, "M2", rt_max, n_rep=args.n_rep)
            errs = c2.tail_errors(tt)
            etests = c2.tail_error_tests(errs)
            dtests = c2.empirical_tail_tests(tt["summary"])
            stests = pd.concat([c2.tail_signed_tests(errs, b) for b in c2.BOUNDS], ignore_index=True)
            save(stests, f"{pre}_tail_signed_tests.csv")
            for k, v in tt.items():
                save(v, f"{pre}_tail_{k}.csv")
            save(errs, f"{pre}_tail_errors.csv")
            save(etests, f"{pre}_tail_error_tests.csv")
            save(dtests, f"{pre}_tail_data_tests.csv")
            plot_tail_diagnostics(tt, title=f"C2 ({name}): RT tail, mice vs fitted M2 with fixed and collapsing bounds",
                                  save_path=os.path.join(fig_dir, f"c2_{name}_tail_diagnostics.png"))
            plot_tail_by_mouse(tt, save_path=os.path.join(fig_dir, f"c2_{name}_tail_by_mouse.png"))
            lines += ["### RT tail (model M2; model RTs from the mice's own trials replayed, cut to the same window)", "",
                      "Model-free descriptive tests on the mice (the hazard test is inflated by truncation at the cutoff; read the data-minus-model tests below instead):", "",
                      _md_table(dtests), "",
                      "Data minus model (cut to the same window), tested against 0 across mice; positive = the mice have a higher late hazard ratio / late accuracy / slow-tail fraction than the model:", "",
                      _md_table(stests), "", "Distance between model and mouse tail, Weibull minus fixed (negative = collapsing closer):", "",
                      _md_table(etests), "",
                      _md_table(errs[["subject_id", "bound", "survival_error", "hazard_ratio_data", "hazard_ratio_model", "p_gt2p5_data",
                                      "p_gt2p5_model", "accuracy_gt2s_data", "accuracy_gt2s_model"]], ".3g"), ""]

        if "M2" in bound_tabs[name]:
            curves = c2.boundary_curves(best, "M2", rt_max, np.linspace(0, rt_max, 101))
            save(curves, f"{pre}_boundary_curves.csv")
            plot_fitted_bounds(curves, title=f"C2 ({name}): fitted Weibull bound, M2",
                               rt_marks={"median RT": float(cohorts[rt_max]["rt"].median()),
                                         "90th pct RT": float(cohorts[rt_max]["rt"].quantile(0.9))},
                               save_path=os.path.join(fig_dir, f"c2_{name}_fitted_bounds.png"))

    # --- cross-variant summaries ------------------------------------------------------------------------------
    m2 = {k: v["M2"] for k, v in bound_tabs.items() if "M2" in v}
    if m2:
        plot_bound_comparison({LABELS[k]: v for k, v in m2.items()}, {LABELS[k]: thresholds[k] for k in m2},
                              title="C2: collapsing vs fixed bound (model M2), per mouse",
                              save_path=os.path.join(fig_dir, "c2_bound_comparison_M2.png"))
        rows = []
        names = list(m2)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append({"variant_a": a, "variant_b": b, **c2.cutoff_consistency(m2[a], m2[b])})
        if rows:
            cons = pd.DataFrame(rows)
            save(cons, "c2_variant_consistency.csv")
            lines += ["## Agreement of the per-mouse delta BIC (Weibull - fixed, model M2) between variants", "", _md_table(cons), ""]
    if cbf_verdicts:
        save(pd.DataFrame([{"variant": v, "bound": b, **d} for v, vd in cbf_verdicts.items() for b, d in vd.items()]), "c2_invariance_verdicts.csv")

    with open(os.path.join(out, "c2_report.md"), "w") as f:
        f.write("\n".join(lines))
    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"analysis written to {out} and {fig_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fit", "analyze", "all", "smoke"])
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--fig-dir", default=None, help="figure directory (default results/figures/c2)")
    p.add_argument("--variant", nargs="+", choices=list(VARIANTS), default=None,
                   help=f"variants to fit (default {' '.join(DEFAULT_VARIANTS)}; tail_5p0 is opt-in)")
    p.add_argument("--models", nargs="+", choices=["M1", "M2", "M3"], default=None, help="override the variant's model list")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--no-polish", action="store_true", help="skip the symmetric fixed<->Weibull polish pass (phases 3-4)")
    p.add_argument("--n-jobs", type=int, default=12, help="parallel fits (each ~350 MB; 12 is safe on 7 GB)")
    p.add_argument("--n-rep", type=int, default=20, help="posterior-predictive replays per mouse in analyze")
    args = p.parse_args(argv)
    args.smoke = args.command == "smoke"
    if args.smoke:
        args.out = os.path.join(args.out, "_smoke")
        args.n_rep = 2
        args.models = args.models or ["M1", "M2", "M3"]
    variants = args.variant or (list(VARIANTS) if args.smoke else list(DEFAULT_VARIANTS))
    if args.command in ("fit", "all", "smoke"):
        for v in variants:
            run_fit_variant(v, args)
    if args.command in ("analyze", "all", "smoke"):
        run_analyze(args)


if __name__ == "__main__":
    main()
