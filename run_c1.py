"""
Command-line runner for Week 3, C1 (prior mechanism): fit M1 (start point z), M2 (drift bias v_b) and
M3 (dynamic bias b(t)) to the 10 IBL mice, build the BIC tables and the model-vs-data CBF overlays.
Author: Parth Dhodapkar's Week 3 role (run by Joshua Koilpillai)

Usage:
    python run_c1.py fit      [--seeds 12345 777] [--n-jobs 12]   # the long step (resumable)
    python run_c1.py analyze                                      # tables, statistics, figures (minutes)
    python run_c1.py all
    python run_c1.py fit --variant no_history --n-jobs 6          # sensitivity: no choice-history drift v_h (~35 min)
    python run_c1.py fit --variant joint --n-jobs 12              # follow-up: combined z + v_b model (~1.5 h)
    python run_c1.py fit --variant wide_bounds --n-jobs 12        # sensitivity: t0 >= 0.02, s_z <= 1.2 (~2 h)
    python run_c1.py smoke                                        # tiny end-to-end check on 3 mice
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from src.optimization.empirical import WIDE_BOUNDS, fit_cohort

COHORT = "data/processed/cohort_rt2p5.parquet"
OUT_DIR = "results/c1"
DEFAULT_SEEDS = (12345, 777)
VARIANT_FILES = {"main": "c1_fits.csv", "no_history": "c1_fits_no_history.csv", "joint": "c1_fits_joint.csv",
                 "wide_bounds": "c1_fits_wide_bounds.csv"}
SMOKE_FIT = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)


def run_fit(args) -> pd.DataFrame:
    cohort = pd.read_parquet(args.cohort)
    no_hist = args.variant == "no_history"
    joint = args.variant == "joint"
    cache = os.path.join(args.out, VARIANT_FILES[args.variant])
    kwargs = dict(SMOKE_FIT) if args.smoke else {}
    if args.variant == "wide_bounds":
        kwargs["param_bounds"] = WIDE_BOUNDS
    if no_hist:      # sensitivity: v_h removed (history is then a pure nuisance in the data, not a regressor)
        kwargs["fit_history"] = False
    subjects = sorted(cohort["subject_id"].unique())[:3] if args.smoke else None
    t = time.time()
    fits = fit_cohort(
        cohort, models=("JOINT",) if joint else ("M1", "M2", "M3"), bounds=("fixed",), rt_max=2.5, n_jobs=args.n_jobs,
        cache_path=cache, subjects=subjects, seeds=args.seeds[:1] if no_hist else args.seeds, **kwargs,
    )
    print(f"fits: {len(fits)} rows in {os.path.relpath(cache)} ({(time.time() - t) / 60:.1f} min this call)")
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
    """Tables, across-mouse statistics, posterior-predictive CBFs and figures from `c1_fits.csv`."""
    import matplotlib
    matplotlib.use("Agg")
    from src.optimization import c1
    from src.visualization.plots_c1 import (
        plot_cbf_overlay_by_mouse, plot_cbf_overlay_cohort, plot_cbf_slope_agreement, plot_delta_bic,
    )

    out = args.out
    fig_dir = args.fig_dir or (os.path.join(out, "figures") if args.smoke else os.path.join("results", "figures", "c1"))
    os.makedirs(fig_dir, exist_ok=True)
    fits = pd.read_csv(os.path.join(out, "c1_fits.csv"))
    cohort = pd.read_parquet(args.cohort)
    shapes_path = os.path.join("results", "empirical", "cbf_slopes.csv")
    shapes = pd.read_csv(shapes_path) if os.path.exists(shapes_path) else None

    def save(df, name):
        df.to_csv(os.path.join(out, name), index=False)

    # --- model comparison -------------------------------------------------------------------
    best = c1.collapse_replicates(fits)
    noise = c1.noise_floor(fits)
    thr = c1.decisive_threshold(noise["sd"])
    bic = c1.bic_table(best, "bic", threshold=thr)
    aic = c1.bic_table(best, "aic", threshold=thr)
    tests = c1.pairwise_tests(bic, "bic")
    subsets = {"all mice": None, "excl. lapse-affected (KS019, KS021, CSH_ZAD_002, NYU-01)": c1.LAPSE_AFFECTED,
               "excl. KS021": ("KS021",)}
    sub_tests = c1.subset_summary(bic, subsets, "bic")
    save(best, "c1_best_fits.csv")
    save(c1.parameter_table(best), "c1_parameters.csv")
    save(c1.at_bound(best), "c1_params_at_bounds.csv")
    save(bic, "c1_bic_table.csv")
    save(aic, "c1_aic_table.csv")
    save(c1.bic_weights(bic), "c1_bic_weights.csv")
    save(tests, "c1_pairwise_tests.csv")
    save(c1.pairwise_tests(aic, "aic"), "c1_pairwise_tests_aic.csv")
    save(sub_tests, "c1_subset_tests.csv")
    save(c1.winner_counts(bic), "c1_winner_counts.csv")
    save(pd.DataFrame([{**noise, "decisive_threshold": thr}]), "c1_noise_floor.csv")

    # --- model-vs-data CBF --------------------------------------------------------------------
    emp = c1.empirical_cbf(cohort)
    pred = c1.predicted_cbf(cohort, best, n_rep=args.n_rep, emp=emp)
    cbf_fit = c1.cbf_fit_table(emp, pred, shapes)
    save(emp, "c1_cbf_empirical.csv")
    save(pred, "c1_cbf_predicted.csv")
    save(cbf_fit, "c1_cbf_fit.csv")
    cbf_tests = pd.concat([c1.cbf_model_tests(cbf_fit, "rmse"), c1.cbf_model_tests(cbf_fit, "chi2_per_bin")], ignore_index=True)
    save(cbf_tests, "c1_cbf_model_tests.csv")
    conc = c1.concordance(bic, shapes, cbf_fit) if shapes is not None else None
    if conc is not None:
        save(conc, "c1_concordance.csv")

    # --- figures ------------------------------------------------------------------------------
    plot_delta_bic(bic, thr, save_path=os.path.join(fig_dir, "c1_delta_bic.png"))
    plot_cbf_overlay_cohort(emp, pred, save_path=os.path.join(fig_dir, "c1_cbf_overlay_cohort.png"))
    plot_cbf_overlay_cohort(emp, pred, exclude=c1.LAPSE_AFFECTED, title="C1: CBF overlay, lapse-affected mice excluded",
                            save_path=os.path.join(fig_dir, "c1_cbf_overlay_cohort_no_lapse_mice.png"))
    plot_cbf_overlay_by_mouse(emp, pred, bic, save_path=os.path.join(fig_dir, "c1_cbf_overlay_by_mouse.png"))
    plot_cbf_slope_agreement(cbf_fit, save_path=os.path.join(fig_dir, "c1_cbf_slope_agreement.png"))

    # --- sensitivities: refits of M1/M2/M3 under a changed model/search setting ----------------------------
    variants = {}
    for key, fname, pretty in (
        ("no_history", "c1_fits_no_history.csv", "WITHOUT the choice-history drift v_h (one seed)"),
        ("wide_bounds", "c1_fits_wide_bounds.csv", "with RELAXED search bounds t0 in [0.02, 0.6], s_z in [0, 1.2]"),
    ):
        path = os.path.join(out, fname)
        if not os.path.exists(path):
            continue
        v_best = c1.collapse_replicates(pd.read_csv(path))
        v_bic = c1.bic_table(v_best, "bic", threshold=thr)
        v_tests = c1.pairwise_tests(v_bic, "bic")
        v_pred = c1.predicted_cbf(cohort, v_best, n_rep=args.n_rep, emp=emp)
        v_fit = c1.cbf_fit_table(emp, v_pred, shapes)
        save(v_bic, f"c1_{key}_bic_table.csv")
        save(v_tests, f"c1_{key}_pairwise_tests.csv")
        save(c1.parameter_table(v_best), f"c1_{key}_parameters.csv")
        save(c1.at_bound(v_best), f"c1_{key}_params_at_bounds.csv")
        save(v_fit, f"c1_{key}_cbf_fit.csv")
        plot_cbf_overlay_cohort(emp, v_pred, title=f"C1 sensitivity ({key}): CBF overlay",
                                save_path=os.path.join(fig_dir, f"c1_cbf_overlay_cohort_{key}.png"))
        variants[key] = (pretty, v_bic, v_tests, v_fit)

    # --- follow-up: JOINT (start point + drift bias) -----------------------------------------------
    joint_path = os.path.join(out, "c1_fits_joint.csv")
    joint_tab = joint_tests = joint_cbf_tests = None
    if os.path.exists(joint_path):
        joint_best = c1.collapse_replicates(pd.read_csv(joint_path))
        joint_tab = c1.joint_comparison(best, joint_best)
        joint_tests = c1.pairwise_tests(joint_tab, "bic", pairs=c1.JOINT_PAIRS)
        j_pred = c1.predicted_cbf(cohort, joint_best, models=("JOINT",), n_rep=args.n_rep, emp=emp)
        j_fit = c1.cbf_fit_table(emp, j_pred, shapes)
        both_fit = pd.concat([cbf_fit, j_fit], ignore_index=True)
        joint_cbf_tests = c1.cbf_model_tests(both_fit, "rmse", pairs=c1.JOINT_PAIRS)
        save(joint_tab, "c1_joint_bic_table.csv")
        save(joint_tests, "c1_joint_pairwise_tests.csv")
        save(joint_cbf_tests, "c1_joint_cbf_tests.csv")
        save(both_fit[both_fit["model"].isin(["M2", "JOINT"])], "c1_joint_cbf_fit.csv")
        save(c1.parameter_table(joint_best), "c1_joint_parameters.csv")
        plot_cbf_overlay_cohort(emp, pd.concat([pred, j_pred], ignore_index=True), models=("M1", "M2", "M3", "JOINT"),
                                title="C1 follow-up: CBF overlay including the combined z + v_b model",
                                save_path=os.path.join(fig_dir, "c1_cbf_overlay_cohort_joint.png"))

    # --- markdown report ----------------------------------------------------------------------
    short = bic[["subject_id", "bic_M1", "bic_M2", "bic_M3", "dbic_M1_vs_M2", "dbic_M3_vs_M2", "winner", "margin", "evidence"]]
    lines = [
        "# C1 results: M1 (start point) vs M2 (drift bias) vs M3 (dynamic bias)", "",
        f"Cohort: {bic['subject_id'].nunique()} mice, RT cutoff 2.5 s, fixed bounds. Fits: {int(fits.groupby(c1.KEYS).ngroups)} "
        f"(mouse, model) x {int(best['n_reps'].max())} replicate seeds; best-BIC replicate kept.", "",
        f"Run-to-run noise: pooled SD of BIC = {noise['sd']:.1f} (median replicate spread {noise['median_spread']:.1f}, "
        f"max {noise['max_spread']:.1f}). A difference is *decisive* if > {thr:.1f} BIC.", "",
        "## Per-mouse BIC (dBIC < 0: better than M2)", "", _md_table(short, ".6g"), "",
        "## Winners", "", _md_table(c1.winner_counts(bic)), "",
        "## Across-mouse tests on BIC (negative mean_delta favours the first model; Holm over 3 pairs)", "",
        _md_table(tests.drop(columns=["sum_delta"], errors="ignore")), "",
        "## Same tests on mouse subsets", "", _md_table(sub_tests[["subset", "pair", "n_in_subset", "mean_delta", "ci_lo", "ci_hi", "wilcoxon_p", "wilcoxon_p_holm", "ttest_p"]]), "",
        "## CBF agreement (RMSE, lower = closer to the data)", "",
        _md_table(cbf_fit.pivot(index="subject_id", columns="model", values="rmse").reset_index(), ".3f"), "",
        _md_table(cbf_tests), "",
    ]
    for key, (pretty, v_bic, v_tests, v_fit) in variants.items():
        lines += [f"## Sensitivity ({key}): refit {pretty}; BIC not comparable with the main table unless the model is unchanged", "",
                  _md_table(v_bic[["subject_id", "bic_M1", "bic_M2", "bic_M3", "dbic_M1_vs_M2", "dbic_M3_vs_M2", "winner"]], ".6g"), "",
                  _md_table(v_tests.drop(columns=["sum_delta"], errors="ignore")), "",
                  _md_table(v_fit.pivot(index="subject_id", columns="model", values="rmse").reset_index(), ".3f"), ""]
    if joint_tab is not None:
        lines += ["## Follow-up: combined start point + drift bias (JOINT, 1 extra parameter) vs M1/M2/M3", "",
                  _md_table(joint_tab[["subject_id", "bic_M2", "bic_JOINT", "dbic_JOINT_vs_M2", "dbic_JOINT_vs_M3", "p_z", "p_v_b", "winner"]], ".6g"), "",
                  _md_table(joint_tests.drop(columns=["sum_delta"], errors="ignore")), "",
                  "CBF discrepancy (RMSE), JOINT vs the others:", "", _md_table(joint_cbf_tests), ""]
    if conc is not None:
        lines += ["## BIC winner vs empirical CBF shape", "", _md_table(conc), ""]
    with open(os.path.join(out, "c1_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"analysis written to {out} and {fig_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fit", "analyze", "all", "smoke"])
    p.add_argument("--cohort", default=COHORT)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--fig-dir", default=None, help="figure directory (default results/figures/c1)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--variant", choices=list(VARIANT_FILES), default="main",
                   help="no_history: refit without the choice-history drift v_h (robustness of the M1/M2/M3 ranking); "
                        "joint: fit the combined start-point + drift-bias model; "
                        "wide_bounds: refit M1/M2/M3 with t0 and s_z search bounds relaxed")
    p.add_argument("--n-jobs", type=int, default=12, help="parallel fits (each ~350 MB; 12 is safe on 7 GB)")
    p.add_argument("--n-rep", type=int, default=20, help="posterior-predictive replays per mouse in analyze")
    args = p.parse_args(argv)
    args.smoke = args.command == "smoke"
    if args.smoke:
        args.out = os.path.join(args.out, "_smoke")
        args.n_rep = 2
    if args.smoke:            # exercise every optional branch of the analysis too
        for variant in ("main", "no_history", "joint", "wide_bounds"):
            args.variant = variant
            run_fit(args)
    elif args.command in ("fit", "all"):
        run_fit(args)
    if args.command in ("analyze", "all", "smoke"):
        run_analyze(args)


if __name__ == "__main__":
    main()
