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

Robust C1 (v2), see README section "Robust C1" and scripts/ada_c1_robust.sbatch:
    python run_c1.py fit --variant lapse  --n-jobs 40     # wide bounds + lapse/contaminant mixture, 3 seeds
    python run_c1.py fit --variant robust --n-jobs 40     # ... + trial-by-trial inferred prior (ideal observer)
    python run_c1.py rescore --variant robust             # common-seed high-precision re-scoring of every fit
    python run_c1.py fit --variant fastguess --n-jobs 38  # robust + prior-biased fast-guess component
    python run_c1.py fit --variant robust_tau --n-jobs 38 # robust M3 with a finer tau_b grid (merged as robust_finetau)
    python run_c1.py analyze                              # also writes c1_<variant>_* tables when those fits exist
    # replication cohort (data/processed/replication/, built by `run_track_b.py --replication`):
    python run_c1.py all --variant robust --cohort data/processed/replication/cohort_rt2p5.parquet \\
        --extended data/processed/replication/cohort_extended.parquet --out results/c1_replication
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from src.optimization.empirical import WIDE_BOUNDS, fit_cohort, rescore_fits

COHORT = "data/processed/cohort_rt2p5.parquet"
EXTENDED = "data/processed/cohort_extended.parquet"
OUT_DIR = "results/c1"
DEFAULT_SEEDS = (12345, 777)
VARIANT_FILES = {"main": "c1_fits.csv", "no_history": "c1_fits_no_history.csv", "joint": "c1_fits_joint.csv",
                 "wide_bounds": "c1_fits_wide_bounds.csv", "lapse": "c1_fits_lapse.csv", "robust": "c1_fits_robust.csv",
                 "fastguess": "c1_fits_fastguess.csv", "robust_tau": "c1_fits_robust_tau.csv"}
SMOKE_FIT = dict(n_screen=2, n_starts=1, n_restarts=0, maxiter=15, n_sim_per_cond=60, n_sim_final=120)

# Robust C1 (v2). Each adds one fix on top of the previous, so their effects can be told apart:
#   wide_bounds (existing) -> lapse: + lapse / contaminant mixture -> robust: + inferred prior.
# Both use 3 replicate seeds and are re-scored on a common seed (`rescore`).
ROBUST_VARIANTS = {
    "lapse": dict(label="wide bounds + lapse/contaminant mixture (true block prior)", inferred=False),
    "robust": dict(label="wide bounds + lapse/contaminant mixture + inferred (ideal-observer) prior", inferred=True),
    "fastguess": dict(label="robust + prior-biased fast-guess (anticipatory) component", inferred=True),
}
# M3 refit with a finer tau_b grid (a superset of the default 0.15 / 0.35 / 0.8), robust settings otherwise; merged with
# the robust M1 / M2 / M3 fits in `analyze` (best of both per mouse), so M3 cannot be handicapped by a coarse grid.
FINE_TAU_GRID = (0.08, 0.15, 0.25, 0.35, 0.5, 0.8, 1.2)
EXTRA_VARIANTS = {"robust_tau": dict(label="robust, M3 with the fine tau_b grid", inferred=True, models=("M3",))}
ROBUST_SEEDS = (12345, 777, 31337)
TAU_SEEDS = (22345, 1777, 41337)   # distinct from ROBUST_SEEDS so merged replicates keep separate keys
RESCORE = dict(n_sim_per_cond=4000, seed=2024)
SMOKE_RESCORE = dict(n_sim_per_cond=150, seed=2024)


def variant_overrides(variant: str) -> dict:
    """fit_ddm keywords that define a variant's model (the same ones are needed to re-score its fits)."""
    kw = {}
    if variant in ("wide_bounds", "lapse", "robust", "fastguess", "robust_tau"):
        kw["param_bounds"] = WIDE_BOUNDS
    if variant in ROBUST_VARIANTS or variant in EXTRA_VARIANTS:
        kw["fit_lapse"] = True
    if variant == "fastguess":
        kw["fit_guess"] = True
    if variant == "robust_tau":
        kw["tau_b_grid"] = FINE_TAU_GRID
    if variant == "no_history":      # sensitivity: v_h removed (history is then a pure nuisance in the data, not a regressor)
        kw["fit_history"] = False
    return kw


def load_cohort(path: str, inferred: bool = False, extended: str = EXTENDED) -> pd.DataFrame:
    """The 2.5 s cohort; with `inferred`, plus `prior_inferred` / `prior_model` (binned) from the ideal observer."""
    cohort = pd.read_parquet(path)
    if inferred:
        from src.data.inferred_prior import attach_inferred_prior
        cohort = attach_inferred_prior(cohort, pd.read_parquet(extended))
    return cohort


def _spec(variant: str) -> dict:
    return ROBUST_VARIANTS.get(variant) or EXTRA_VARIANTS.get(variant) or {}


def fit_frame(cohort: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The table the fitter sees: for the inferred-prior variant the model's `prior` covariate is `prior_model`."""
    if _spec(variant).get("inferred"):
        return cohort.assign(prior=cohort["prior_model"])
    return cohort


def _seeds(args) -> list:
    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(TAU_SEEDS if args.variant == "robust_tau" else
                     ROBUST_SEEDS if args.variant in ROBUST_VARIANTS else DEFAULT_SEEDS)
    if args.smoke:
        seeds = seeds[:2]
    return seeds[:1] if args.variant == "no_history" else seeds


def run_fit(args) -> pd.DataFrame:
    cohort = load_cohort(args.cohort, _spec(args.variant).get("inferred", False), args.extended)
    joint = args.variant == "joint"
    cache = os.path.join(args.out, VARIANT_FILES[args.variant])
    kwargs = dict(SMOKE_FIT) if args.smoke else {}
    kwargs.update(variant_overrides(args.variant))
    subjects = sorted(cohort["subject_id"].unique())[:3] if args.smoke else None
    t = time.time()
    fits = fit_cohort(
        fit_frame(cohort, args.variant), models=("JOINT",) if joint else _spec(args.variant).get("models", ("M1", "M2", "M3")),
        bounds=("fixed",), rt_max=2.5,
        n_jobs=args.n_jobs, cache_path=cache, subjects=subjects, seeds=_seeds(args), **kwargs,
    )
    print(f"fits: {len(fits)} rows in {os.path.relpath(cache)} ({(time.time() - t) / 60:.1f} min this call)")
    return fits


def rescored_path(out: str, variant: str) -> str:
    return os.path.join(out, VARIANT_FILES[variant].replace(".csv", "_rescored.csv"))


def run_rescore(args) -> pd.DataFrame:
    """Re-score every fit of a variant with one common-seed, high-precision likelihood evaluation."""
    fits_path = os.path.join(args.out, VARIANT_FILES[args.variant])
    fits = pd.read_csv(fits_path)
    cohort = load_cohort(args.cohort, _spec(args.variant).get("inferred", False), args.extended)
    t = time.time()
    res = rescore_fits(fit_frame(cohort, args.variant), fits, n_jobs=args.n_jobs, cache_path=rescored_path(args.out, args.variant),
                       **(SMOKE_RESCORE if args.smoke else RESCORE), **variant_overrides(args.variant))
    print(f"rescored: {len(res)} rows in {os.path.relpath(rescored_path(args.out, args.variant))} ({(time.time() - t) / 60:.1f} min)")
    return res


def _md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)




def _load_variant_fits(out: str, key: str):
    """(fits, rescored?) for a variant: the re-scored table when every fit has been re-scored, else the raw fits."""
    fpath, rpath = os.path.join(out, VARIANT_FILES[key]), rescored_path(out, key)
    if not os.path.exists(fpath):
        return None, False
    fits = pd.read_csv(fpath)
    if os.path.exists(rpath):
        r = pd.read_csv(rpath)
        if len(r) >= len(fits):
            return r, True
        print(f"[{key}] only {len(r)}/{len(fits)} fits re-scored; using the fits' own BIC (run `rescore`)")
    return fits, False


def analyze_robust(args, out, fig_dir, cohort, emp, shapes, subsets, save):
    """Tables, tests, CBF checks and figures for every robust variant whose fits exist; returns (results, prior_cmp)."""
    from src.optimization import c1
    from src.visualization.plots_c1 import plot_cbf_overlay_by_mouse, plot_cbf_overlay_cohort, plot_delta_bic

    robust, cohort_inf, prior_cmp = {}, None, None
    if os.path.exists(args.extended):
        cohort_inf = load_cohort(args.cohort, inferred=True, extended=args.extended)
        prior_cmp = c1.prior_covariate_comparison(cohort_inf)
        save(prior_cmp, "c1_prior_covariate_comparison.csv")

    jobs = []
    for key, spec in ROBUST_VARIANTS.items():
        fits, rescored = _load_variant_fits(out, key)
        if fits is not None:
            jobs.append((key, spec, fits, rescored, None))
    # robust + fine tau_b grid for M3: best of both M3 fits per mouse; noise floor from the robust fits alone
    rob, rob_r = _load_variant_fits(out, "robust")
    tau, tau_r = _load_variant_fits(out, "robust_tau")
    if rob is not None and tau is not None:
        jobs.append(("robust_finetau", dict(label="robust, M3 = best of the default and the fine tau_b grid", inferred=True),
                     pd.concat([rob, tau], ignore_index=True), rob_r and tau_r, c1.noise_floor(rob)))

    for key, spec, v_fits, rescored, noise_override in jobs:
        v_cohort = cohort_inf if spec["inferred"] else cohort
        if v_cohort is None:
            print(f"[{key}] needs {args.extended} for the inferred prior; skipped")
            continue
        pcol = "prior_model" if spec["inferred"] else "prior"
        v_best = c1.collapse_replicates(v_fits)
        v_noise = noise_override or c1.noise_floor(v_fits)
        v_thr = c1.decisive_threshold(v_noise["sd"])
        v_bic = c1.bic_table(v_best, "bic", threshold=v_thr)
        v_tests = c1.pairwise_tests(v_bic, "bic")
        v_sub = c1.subset_summary(v_bic, subsets, "bic")
        v_pred = c1.predicted_cbf(v_cohort, v_best, n_rep=args.n_rep, emp=emp, prior_col=pcol)
        v_fit = c1.cbf_fit_table(emp, v_pred, shapes)
        v_cbf_tests = c1.cbf_model_tests(v_fit, "rmse")
        boot = c1.cbf_slope_bootstrap(v_cohort, v_best, n_boot=args.n_boot, emp=emp, prior_col=pcol)
        lr, lr_sum = c1.cbf_slope_lr(boot)
        save(v_best, f"c1_{key}_best_fits.csv")
        save(v_bic, f"c1_{key}_bic_table.csv")
        save(v_tests, f"c1_{key}_pairwise_tests.csv")
        save(v_sub, f"c1_{key}_subset_tests.csv")
        save(c1.winner_counts(v_bic), f"c1_{key}_winner_counts.csv")
        save(pd.DataFrame([{**v_noise, "decisive_threshold": v_thr, "rescored": rescored}]), f"c1_{key}_noise_floor.csv")
        save(c1.parameter_table(v_best), f"c1_{key}_parameters.csv")
        save(c1.at_bound(v_best, bounds=WIDE_BOUNDS), f"c1_{key}_params_at_bounds.csv")
        save(v_pred, f"c1_{key}_cbf_predicted.csv")
        save(v_fit, f"c1_{key}_cbf_fit.csv")
        save(v_cbf_tests, f"c1_{key}_cbf_model_tests.csv")
        save(boot, f"c1_{key}_cbf_slope_bootstrap.csv")
        save(pd.concat([lr, pd.DataFrame([{**lr_sum, "subject_id": "ALL"}])], ignore_index=True), f"c1_{key}_cbf_slope_lr.csv")
        plot_delta_bic(v_bic, v_thr, save_path=os.path.join(fig_dir, f"c1_{key}_delta_bic.png"))
        plot_cbf_overlay_cohort(emp, v_pred, title=f"C1 {key}: CBF overlay ({spec['label']})",
                                save_path=os.path.join(fig_dir, f"c1_cbf_overlay_cohort_{key}.png"))
        plot_cbf_overlay_by_mouse(emp, v_pred, v_bic, save_path=os.path.join(fig_dir, f"c1_cbf_overlay_by_mouse_{key}.png"))
        robust[key] = dict(label=spec["label"], fits=v_fits, noise=v_noise, thr=v_thr, bic=v_bic, tests=v_tests, sub=v_sub,
                           fit=v_fit, cbf_tests=v_cbf_tests, boot=boot, lr=lr, lr_sum=lr_sum, rescored=rescored, best=v_best)
    return robust, prior_cmp


def robust_report_lines(robust, prior_cmp) -> list:
    from src.optimization import c1
    lines = []
    if prior_cmp is not None:
        lines += ["## Which prior do the choices follow? (model-free logistic regression, same parameter count)", "",
                  "Positive dloglik = the ideal-observer (inferred) prior explains the choices better than the true block label.", "",
                  _md_table(prior_cmp, ".6g"), ""]
    for key, r in robust.items():
        par = c1.parameter_table(r["best"])
        mix_cols = [c for c in ("p_lapse", "p_lapse_side", "p_lapse_prior", "p_guess", "p_guess_width", "p_guess_prior") if c in par]
        n_seeds = int(r["fits"].groupby(c1.KEYS).size().max())
        lines += [f"## Robust C1 ({key}): {r['label']}", "",
                  f"{int(r['fits'].groupby(c1.KEYS).ngroups)} (mouse, model) fits x up to {n_seeds} seeds; "
                  f"BIC {'re-scored on a common seed (CRN across models)' if r['rescored'] else 'from each fit (NOT re-scored)'}. "
                  f"Pooled run-to-run SD {r['noise']['sd']:.1f}; decisive if > {r['thr']:.1f}. "
                  "BIC is comparable across models WITHIN this section only.", "",
                  _md_table(r["bic"][["subject_id", "bic_M1", "bic_M2", "bic_M3", "dbic_M1_vs_M2", "dbic_M3_vs_M2", "winner", "margin", "evidence"]], ".6g"), "",
                  _md_table(c1.winner_counts(r["bic"])), "",
                  _md_table(r["tests"].drop(columns=["sum_delta"], errors="ignore")), ""]
        if len(r["sub"]):
            lines += [_md_table(r["sub"][["subset", "pair", "n_in_subset", "mean_delta", "ci_lo", "ci_hi", "wilcoxon_p", "wilcoxon_p_holm"]]), ""]
        lines += ["CBF agreement (RMSE):", "", _md_table(r["fit"].pivot(index="subject_id", columns="model", values="rmse").reset_index(), ".3f"), "",
                  _md_table(r["cbf_tests"]), "",
                  "Calibrated CBF-slope test (parametric bootstrap; loglr > 0 favours M2; fitted parameters treated as known):", "",
                  _md_table(r["boot"][["subject_id", "model", "slope_obs", "slope_null_mean", "slope_null_sd", "p_two_sided", "z"]], ".3g"), "",
                  _md_table(r["lr"], ".3g"), "", _md_table(pd.DataFrame([r["lr_sum"]]), ".3g"), ""]
        if mix_cols:
            lines += ["Mixture (lapse / fast-guess) parameters:", "", _md_table(par[["subject_id", "model"] + mix_cols], ".3g"), ""]
    return lines


def run_analyze_robust_only(args) -> None:
    """Analysis when the output folder holds only robust-variant fits (e.g. the replication cohort)."""
    import matplotlib
    matplotlib.use("Agg")
    from src.optimization import c1
    out = args.out
    fig_dir = args.fig_dir or os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cohort = pd.read_parquet(args.cohort)

    def save(df, name):
        df.to_csv(os.path.join(out, name), index=False)

    present = set(cohort["subject_id"].unique())
    subsets = {"all mice": None}
    emp = c1.empirical_cbf(cohort)
    save(emp, "c1_cbf_empirical.csv")
    robust, prior_cmp = analyze_robust(args, out, fig_dir, cohort, emp, None, subsets, save)
    lines = [f"# C1 results (robust variants only): {len(present)} mice from {args.cohort}", ""]
    lines += robust_report_lines(robust, prior_cmp)
    with open(os.path.join(out, "c1_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"analysis written to {out} and {fig_dir}")


def run_analyze(args) -> None:
    """Tables, across-mouse statistics, posterior-predictive CBFs and figures from `c1_fits.csv`."""
    import matplotlib
    matplotlib.use("Agg")
    from src.optimization import c1
    from src.visualization.plots_c1 import (
        plot_cbf_overlay_by_mouse, plot_cbf_overlay_cohort, plot_cbf_slope_agreement, plot_delta_bic,
    )

    out = args.out
    if not os.path.exists(os.path.join(out, "c1_fits.csv")):
        return run_analyze_robust_only(args)
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
        save(c1.at_bound(v_best, bounds=WIDE_BOUNDS if key == "wide_bounds" else None), f"c1_{key}_params_at_bounds.csv")
        save(v_fit, f"c1_{key}_cbf_fit.csv")
        plot_cbf_overlay_cohort(emp, v_pred, title=f"C1 sensitivity ({key}): CBF overlay",
                                save_path=os.path.join(fig_dir, f"c1_cbf_overlay_cohort_{key}.png"))
        variants[key] = (pretty, v_bic, v_tests, v_fit)

    # --- robust C1 (v2): lapse mixture, inferred prior, 3 seeds, common-seed re-scoring -------------------
    robust, prior_cmp = analyze_robust(args, out, fig_dir, cohort, emp, shapes, subsets, save)

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
    lines += robust_report_lines(robust, prior_cmp)
    if conc is not None:
        lines += ["## BIC winner vs empirical CBF shape", "", _md_table(conc), ""]
    with open(os.path.join(out, "c1_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"analysis written to {out} and {fig_dir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fit", "rescore", "analyze", "all", "smoke"])
    p.add_argument("--cohort", default=COHORT)
    p.add_argument("--extended", default=EXTENDED, help="cohort_extended.parquet (session order, feedback) for the inferred prior")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--fig-dir", default=None, help="figure directory (default results/figures/c1)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"replicate seeds (default {list(DEFAULT_SEEDS)}; {list(ROBUST_SEEDS)} for the lapse / robust variants)")
    p.add_argument("--variant", choices=list(VARIANT_FILES), default="main",
                   help="no_history: refit without the choice-history drift v_h (robustness of the M1/M2/M3 ranking); "
                        "joint: fit the combined start-point + drift-bias model; "
                        "wide_bounds: refit M1/M2/M3 with t0 and s_z search bounds relaxed; "
                        "lapse: wide bounds + lapse/contaminant mixture; robust: lapse + inferred prior; "
                        "fastguess: robust + prior-biased fast guesses; robust_tau: robust M3 with a finer tau_b grid")
    p.add_argument("--n-jobs", type=int, default=12, help="parallel fits (each ~350 MB; 12 is safe on 7 GB)")
    p.add_argument("--n-rep", type=int, default=20, help="posterior-predictive replays per mouse in analyze")
    p.add_argument("--n-boot", type=int, default=200, help="parametric-bootstrap replays per (mouse, model) for the CBF-slope test")
    args = p.parse_args(argv)
    args.smoke = args.command == "smoke"
    if args.smoke:
        args.out = os.path.join(args.out, "_smoke")
        args.n_rep = 2
    if args.smoke:
        args.n_boot = 5
    if args.smoke:            # exercise every optional branch of the analysis too
        for variant in ("main", "no_history", "joint", "wide_bounds", "lapse", "robust", "fastguess", "robust_tau"):
            args.variant = variant
            run_fit(args)
            if variant in ROBUST_VARIANTS or variant in EXTRA_VARIANTS:
                run_rescore(args)
    elif args.command in ("fit", "all"):
        run_fit(args)
        if args.command == "all" and (args.variant in ROBUST_VARIANTS or args.variant in EXTRA_VARIANTS):
            run_rescore(args)
    elif args.command == "rescore":
        run_rescore(args)
    if args.command in ("analyze", "all", "smoke"):
        run_analyze(args)


if __name__ == "__main__":
    main()
