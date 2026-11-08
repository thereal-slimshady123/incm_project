"""Regenerates notebooks/02_track_b_empirical_baseline.ipynb (run `python run_track_b.py --empirical` to execute it)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Track B — Empirical Baseline: IBL Mouse Behaviour in Biased Blocks
**Author:** Joshua Koilpillai (2024101128) · **Project:** How does an uncued, inferred prior enter the decision process? (INCM)

Week 2 deliverable (plan §4, Track B): empirical **psychometric** and **chronometric** curves, the diagnostic
**Conditional Bias Function (CBF)**, and a **heterogeneity profile** for the 10 IBL mice built in Week 1, plus a
validation of the tools on synthetic mock data and on Track A's simulator (§3.1 contract).

**Conventions** (Week 1): `prior = +1` right-biased block (P(left stimulus) = 0.2); `choice = +1` Right response;
`contrast` signed (+ = right); **`rt = response_times − stimOn`** (stimulus onset to wheel-threshold crossing), kept in [0.1, 2.5] s.

> **RT definition (changed after the Week 2 audit).** The Week 1 cohort used `firstMovement − stimOn`. That value is ≤ 0.1 s on
> ~45% of responded trials (anticipatory wheel movement), so the 0.1 s filter discarded nearly half of the data and — as §4.2
> shows — those trials carry more of the prior, which distorted the CBF. The contract `rt` is now the response time; the
> first-movement time is kept as `rt_firstmove` for sensitivity analyses. Same 10 mice, many more trials.

Everything below is **descriptive**: the C1 adjudication between M1/M2/M3 (ΔBIC + model-predicted CBFs) happens in Week 3.
""")

code(r"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.empirical import *
from src.empirical.cbf import compute_cbf, DEFAULT_MIN_ABS_SLOPE
from src.visualization import *
from src.models.simulator import simulate_ddm

N_BOOT = 500          # bootstrap resamples (CBF slopes, prior shift)
TAB = ROOT / "results" / "empirical"; FIG = ROOT / "results" / "figures" / "empirical"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
plt.rcParams["figure.dpi"] = 110

df = pd.read_parquet(ROOT / "data/processed/cohort_rt2p5.parquet")
ext = pd.read_parquet(ROOT / "data/processed/cohort_extended.parquet")
ext = ext[ext["rt_in_2p5"]].reset_index(drop=True)            # same trials as df, plus session_eid, block_pos, rt_firstmove
CL = "session_eid"                                            # bootstrap whole sessions (trials within a session are dependent)
assert len(ext) == len(df) and np.allclose(ext["rt"], df["rt"])
summary = pd.read_csv(ROOT / "data/processed/summary_stats.csv")
print(f"{len(df):,} trials, {df.subject_id.nunique()} mice; columns: {list(df.columns)}")
""")

md("## 1. Dataset overview")
code(r"""
ov = summary[summary.rt_max == 2.5][["subject_id", "lab", "n_sessions", "n_raw_trials", "n_trials",
                                      "dropped_rt_too_fast", "dropped_rt_too_slow", "p_right",
                                      "easy_accuracy", "rt_median"]].round(3)
ov["frac_rt_too_fast"] = (ov["dropped_rt_too_fast"] / ov["n_raw_trials"]).round(3)
ov.to_csv(TAB / "dataset_overview.csv", index=False)
ov
""")

md(r"""
## 2. Psychometric functions
P(Right) vs signed contrast for each block prior, with a lapse-corrected logistic
$P(R\mid c)=\gamma+(1-\gamma-\lambda)\,\sigma\!\big((c-\mu)/s\big)$ fitted by maximum likelihood to each mouse and block
(μ = point of subjective equality, s = slope scale; γ = P(Right) for strongly *left* stimuli, i.e. the error rate on left stimuli;
λ = P(Left) for strongly *right* stimuli).
""")
code(r"""
plot_psychometric_chronometric_cohort(df, save_path=str(FIG / "cohort_psychometric_chronometric.png"));
""")
code(r"""
fits = fit_psychometric_by(df, n_boot=0)
fits.to_csv(TAB / "psychometric_fits.csv", index=False)
psychometric_table(df).to_csv(TAB / "psychometric_table.csv", index=False)
plot_psychometric_by_mouse(df, fits=fits, save_path=str(FIG / "psychometric_by_mouse.png"));
print("all fits converged:", bool(fits["success"].all()))
fits.round(3)
""")
md(r"""
### 2.1 Prior shift
`shift = μ(left block) − μ(right block)` in contrast units: positive means the psychometric curve moves toward the block's
likelier side. Bootstrap 95% intervals resample **whole sessions** (500 resamples), because trials within a session are not independent;
the trial-level intervals are kept for comparison (`*_trial`). The per-block lapse parameters are reported too: the prior can also move
the asymptotes (γ, λ), not only the horizontal position μ.
""")
code(r"""
ps_trial = prior_shift(df, n_boot=N_BOOT, seed=1)
ps = prior_shift(ext, n_boot=N_BOOT, seed=1, cluster_col=CL)
ps = ps.merge(ps_trial[["subject_id", "shift_lo", "shift_hi"]].rename(columns={"shift_lo": "shift_lo_trial", "shift_hi": "shift_hi_trial"}), on="subject_id")
ps.to_csv(TAB / "prior_shift.csv", index=False)
n_sig = int((ps["shift_lo"] > 0).sum())
display(Markdown(f"**{n_sig}/{len(ps)} mice** have a prior shift whose 95% interval excludes 0; all {int((ps['shift']>0).sum())} "
                 f"shifts are positive. Mean shift = {ps['shift'].mean():.3f} contrast units "
                 f"(range {ps['shift'].min():.3f}–{ps['shift'].max():.3f}); mean ΔP(Right) at 0% contrast = {ps['prior_effect_c0'].mean():.3f}."))
display(Markdown("**Block-dependent lapses** (right-block vs left-block γ and λ; a large difference means the prior moves the asymptote):"))
display(pd.concat([ps[["subject_id", "shift", "shift_lo", "shift_hi", "shift_lo_trial", "shift_hi_trial"]],
                   ps[["gamma_right_block", "gamma_left_block", "lam_right_block", "lam_left_block"]]], axis=1).round(3))
""")

md(r"""
## 3. Chronometric functions
Mean RT vs signed contrast per block, per-mouse panels, RT distributions, and the correct-vs-error split
(informative for DDMs: a starting-point bias predicts *fast* errors toward the prior, a drift bias does not).
""")
code(r"""
chrono = chronometric_table(df)
chrono.to_csv(TAB / "chronometric_table.csv", index=False)
plot_chronometric_by_mouse(df, save_path=str(FIG / "chronometric_by_mouse.png"));
""")
code(r"""
plot_rt_distributions(df, save_path=str(FIG / "rt_distributions.png"))
rtq = rt_quantile_table(df); rtq.to_csv(TAB / "rt_quantiles_by_contrast.csv", index=False)

by_out = chronometric_table(df, by=("subject_id",), by_outcome=True)
by_out.to_csv(TAB / "chronometric_by_outcome.csv", index=False)
co = by_out.assign(abs_c=by_out.contrast.abs()).groupby(["correct", "abs_c"])["mean_rt"].mean().unstack(0)
co.columns = ["error", "correct"]; co["error − correct (s)"] = co["error"] - co["correct"]
display(Markdown("**Mean RT (s) by |contrast| and outcome (mean over mice):**"))
display(co.round(3))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
caf = accuracy_by_rt_bin(df); caf.to_csv(TAB / "conditional_accuracy.csv", index=False)
for _, g in caf.groupby("subject_id"):
    axes[0].plot(g.bin + 1, g.accuracy, color="#888", alpha=.5)
m = caf.groupby("bin")["accuracy"].mean(); axes[0].plot(m.index + 1, m.values, "k-o", lw=2)
axes[0].set(xlabel="RT quantile bin (1 = fastest)", ylabel="Accuracy (non-zero contrast)", title="Conditional accuracy function")
mc = chrono.assign(abs_c=chrono.contrast.abs()).groupby(["subject_id", "abs_c"])["mean_rt"].mean().reset_index()
for _, g in mc.groupby("subject_id"):
    axes[1].plot(g.abs_c, g.mean_rt, color="#888", alpha=.5)
mm = mc.groupby("abs_c")["mean_rt"].mean(); axes[1].plot(mm.index, mm.values, "k-o", lw=2)
axes[1].set(xlabel="|contrast|", ylabel="Mean RT (s)", title="Chronometric function vs |contrast|")
fig.tight_layout(); fig.savefig(FIG / "conditional_accuracy_and_chrono_abs.png", dpi=200, bbox_inches="tight")
""")

md(r"""
## 4. Conditional Bias Function (CBF)
$\mathrm{CBF}(\text{bin})=P(\text{Right}\mid \text{prior Right},\text{RT bin})-P(\text{Right}\mid \text{prior Left},\text{RT bin})$ over
within-mouse **RT quintiles**. Predicted shapes: **M1 starting point → falling**, **M2 drift bias → flat**, **M3 dynamic bias → rising**.

*Design note.* In the IBL task 80% of stimuli appear on the block's likelier side, so the raw difference mixes stimulus statistics
with the mouse's bias. The CBF is therefore computed **within each signed contrast** and averaged with inverse-variance weights,
on ambiguous stimuli (|contrast| ≤ 0.125) by default. A per-mouse shape label needs the (session-level) bootstrap CI of the slope to exclude 0
**and** |slope| ≥ 0.03 per bin (calibrated so that Track A's pure drift-bias model M2, slope ≈ −0.027, is labelled flat; see §6.1).
""")
code(r"""
curves, slopes = cbf_by_subject(ext, n_boot=N_BOOT, seed=7, cluster_col=CL)
_, slopes_trial = cbf_by_subject(df, n_boot=N_BOOT, seed=7)          # trial-level bootstrap, for comparison
curves.to_csv(TAB / "cbf_curves.csv", index=False); slopes.merge(slopes_trial[["subject_id", "slope_lo", "slope_hi", "shape"]].rename(
    columns={"slope_lo": "slope_lo_trial", "slope_hi": "slope_hi_trial", "shape": "shape_trial"}), on="subject_id").to_csv(TAB / "cbf_slopes.csv", index=False)
cohort = cohort_cbf(curves); cohort.to_csv(TAB / "cbf_cohort.csv", index=False)
test = cohort_slope_test(slopes)
pd.Series({k: v for k, v in test.items()}).to_frame("value").to_csv(TAB / "cbf_cohort_slope_test.csv")
display(Markdown(f"Per-mouse shapes with session-level bootstrap: {slopes['shape'].value_counts().to_dict()}; "
                 f"with trial-level bootstrap: {slopes_trial['shape'].value_counts().to_dict()}."))

# RT bin edges per mouse: Week 3 applies these same edges to model simulations
edges = pd.DataFrame({s: np.r_[compute_cbf(g)["rt_lo"].iloc[0], compute_cbf(g)["rt_hi"].to_numpy()]
                      for s, g in df.groupby("subject_id")}).T
edges.columns = [f"edge{i}" for i in range(edges.shape[1])]; edges.index.name = "subject_id"
edges.to_csv(TAB / "cbf_bin_edges.csv")

plot_cbf_by_mouse(curves, slopes, save_path=str(FIG / "cbf_by_mouse.png"));
""")
code(r"""
plot_cbf_cohort(curves, save_path=str(FIG / "cbf_cohort.png"));
display(Markdown(
    f"**Cohort CBF slope** = {test['mean_slope']:.3f} ± {test['sem_slope']:.3f} per RT bin "
    f"(t-test p = {test['t_p']:.2f}, Wilcoxon p = {test['wilcoxon_p']:.2f}); per-mouse shapes: {test['shape_counts']}."))
slopes.round(3)
""")
md("### 4.1 Sensitivity to the contrast selection")
code(r"""
settings = {"0% contrast only": {"max_abs_contrast": 0.0},
            "|c| ≤ 0.125 (default)": {"max_abs_contrast": 0.125},
            "all contrasts": {"max_abs_contrast": 1.0}}
plot_cbf_sensitivity(df, settings, n_boot=100, save_path=str(FIG / "cbf_sensitivity_contrast.png"));
rows = []
for label, kw in settings.items():
    _, sl = cbf_by_subject(ext, n_boot=200, seed=3, cluster_col=CL, **kw)
    t = cohort_slope_test(sl)
    rows.append({"selection": label, "mean_slope": t["mean_slope"], "sem": t["sem_slope"], "t_p": t["t_p"], **sl["shape"].value_counts().to_dict()})
sens_c = pd.DataFrame(rows).fillna(0); sens_c.to_csv(TAB / "cbf_sensitivity_contrast.csv", index=False); sens_c.round(3)
""")
md(r"""
### 4.2 Sensitivity to the RT definition and to the Week 1 trial exclusion
The Week 1 cohort defined RT as first wheel movement and discarded trials below 0.1 s. Here the CBF is recomputed on the *same*
mice with (a) the response time (current definition), (b) first-movement time on every trial where it is positive, and
(c) the Week 1 definition (first movement, ≥ 0.1 s). If (c) differs from (a) and (b), the exclusion — not the RT variable — drives the difference.
""")
code(r"""
fm = ext["rt_firstmove"]
variants = {
    "response time (current)": (ext, "rt"),
    "first movement, all positive": (ext[fm > 0], "rt_firstmove"),
    "first movement ≥ 0.1 s (Week 1)": (ext[fm >= 0.1], "rt_firstmove"),
}
rows, curves_v, slopes_v = [], {}, {}
for lab, (d, col) in variants.items():
    cu, sl = cbf_by_subject(d, n_boot=200, seed=7, rt_col=col, cluster_col=CL)
    t = cohort_slope_test(sl)
    curves_v[lab], slopes_v[lab] = cu, sl
    rows.append({"RT definition": lab, "n_trials": len(d), "mean_slope": t["mean_slope"], "sem": t["sem_slope"], "t_p": t["t_p"],
                 **sl["shape"].value_counts().to_dict()})
sens_rt = pd.DataFrame(rows).fillna(0); sens_rt.to_csv(TAB / "cbf_sensitivity_rt_definition.csv", index=False)
curves_r, slopes_r = curves_v["first movement ≥ 0.1 s (Week 1)"], slopes_v["first movement ≥ 0.1 s (Week 1)"]
curves_r.to_csv(TAB / "cbf_curves_firstmove_week1.csv", index=False); slopes_r.to_csv(TAB / "cbf_slopes_firstmove_week1.csv", index=False)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for (lab, cur), col in zip(curves_v.items(), ("#111111", "#2ca02c", "#d62728")):
    c = cohort_cbf(cur); ax.errorbar(c.bin + 1, c.cbf, yerr=c["sem"], marker="o", lw=2, capsize=3, color=col, label=lab)
ax.axhline(0, color="gray", ls=":"); ax.set(xlabel="RT quantile bin (1 = fastest)", ylabel="ΔP(Right | prior)", title="Cohort CBF under three RT definitions")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "cbf_sensitivity_rt_definition.png", dpi=200)

# Are the trials with anticipatory first movement (< 0.1 s) a random subset?  Compare how often the choice follows the block.
follows = (ext["choice"] == ext["prior"])
early = fm < 0.1
sel = pd.Series({"P(choice follows block), first movement < 0.1 s": follows[early].mean(),
                 "P(choice follows block), first movement ≥ 0.1 s": follows[fm >= 0.1].mean(),
                 "fraction of trials with first movement < 0.1 s or missing": float((early | fm.isna()).mean())}).round(3)
display(Markdown("**The Week 1 exclusion removed a non-random subset:**")); display(sel.to_frame("value"))
t_r = cohort_slope_test(slopes_r)
sens_rt.round(3)
""")
md(r"""
### 4.3 Robustness: block position and leaving out KS021
The CBF uses the block label as the prior, but the mouse needs some trials to infer a new block. Dropping the first *n* trials of each block
checks that early, mislabelled trials do not drive the result. KS021 is the weakest mouse (large lapses); the second table repeats the cohort test without it.
""")
code(r"""
rows = []
for n0 in (0, 10, 20, 40):
    d = ext[ext["block_pos"] >= n0]
    _, sl = cbf_by_subject(d, n_boot=200, seed=5, cluster_col=CL)
    t = cohort_slope_test(sl)
    rows.append({"skip first n trials of each block": n0, "n_trials": len(d), "mean_slope": t["mean_slope"], "sem": t["sem_slope"], "t_p": t["t_p"],
                 **sl["shape"].value_counts().to_dict()})
sens_blk = pd.DataFrame(rows).fillna(0); sens_blk.to_csv(TAB / "cbf_sensitivity_block_position.csv", index=False)
loo = []
for lab, ex in (("all 10 mice", []), ("without KS021", ["KS021"])):
    t = cohort_slope_test(slopes, exclude=ex); k = ps[~ps.subject_id.isin(ex)]
    loo.append({"cohort": lab, "n_mice": t["n_mice"], "mean_cbf_slope": t["mean_slope"], "sem": t["sem_slope"], "t_p": t["t_p"],
                "mean_prior_shift": k["shift"].mean(), "mean_dP_c0": k["prior_effect_c0"].mean()})
loo = pd.DataFrame(loo); loo.to_csv(TAB / "robustness_without_ks021.csv", index=False)
display(sens_blk.round(3)); loo.round(3)
""")

md(r"""
## 5. Individual-mouse heterogeneity
One row of behavioural metrics per mouse: sensitivity (σ, lapses, accuracy), prior use (shift, ΔP at 0% contrast, prior effect on each lapse),
choice history (the history effect is estimated *within* each block so it cannot be confounded with the block prior),
speed (RT median/IQR/q90, fraction slow, chronometric slope) and the CBF slope. Outliers use a robust z-score (median/MAD) > 2.
""")
code(r"""
prof = mouse_profile(ext, n_boot=N_BOOT, seed=11, shifts=ps, slopes=slopes)
prof.to_csv(TAB / "mouse_profile.csv", index=False)
hsum = heterogeneity_summary(prof); hsum.to_csv(TAB / "heterogeneity_summary.csv", index=False)
corr = profile_correlations(prof); corr.to_csv(TAB / "profile_correlations.csv")
plot_heterogeneity(prof, save_path=str(FIG / "heterogeneity.png"));
hsum.round(3)
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
fig.colorbar(im, label="Spearman ρ (n = 10 mice)"); ax.set_title("Between-mouse metric correlations")
fig.tight_layout(); fig.savefig(FIG / "profile_correlations.png", dpi=200)
strong = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().loc[lambda s: s.abs() > 0.7].sort_values())
display(Markdown("**Strongest between-mouse correlations (|ρ| > 0.7; descriptive, n = 10, no multiple-comparison correction):**"))
strong.round(2).to_frame("rho")
""")
code(r"""
prof[["subject_id", "n_trials", "sigma", "gamma", "lam", "gamma_block_diff", "lam_block_diff", "prior_shift", "prior_effect_c0",
      "history_effect_c0", "rt_median", "rt_q90", "frac_slow", "cbf_slope", "cbf_shape"]].round(3)
""")

md(r"""
## 6. Tool validation on synthetic mock data (plan §4, Week 2 task 4)
`src.empirical.mock` generates §3.1 tables with **known** psychometric parameters, prior strength and RT-dependence of the bias —
independently of Track A's simulator. The tools must recover the truth. The automated version of these checks lives in
`tests/test_empirical.py` (`pytest tests/test_empirical.py`), which also feeds the tools shuffled / extra-column / categorical
variants, single-block tables, Track A simulator output and the real cohort, and checks they reject schema-violating tables.
""")
code(r"""
mc = make_mock_cohort(n_subjects=8, n_trials=10000, seed=21)
truth = mock_truth(mc).set_index("subject_id")
mfit = fit_psychometric_by(mc).groupby("subject_id")[["sigma", "gamma", "lam"]].mean()
mps = prior_shift(mc, n_boot=100).set_index("subject_id")
rec = pd.DataFrame({"sigma_true": truth.sigma, "sigma_fit": mfit.sigma,
                    "lapse_L_true": truth.gamma, "lapse_L_fit": mfit.gamma,
                    "lapse_R_true": truth.lam, "lapse_R_fit": mfit.lam,
                    "shift_true": truth["shift"], "shift_fit": mps["shift"]})
rec.to_csv(TAB / "mock_recovery.csv")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, (a, b, name) in zip(axes, [("sigma_true", "sigma_fit", "σ"), ("lapse_L_true", "lapse_L_fit", "γ (P(R), far left)"),
                                   ("lapse_R_true", "lapse_R_fit", "λ (P(L), far right)"), ("shift_true", "shift_fit", "prior shift")]):
    ax.scatter(rec[a], rec[b], color="k"); lim = [min(rec[a].min(), rec[b].min()), max(rec[a].max(), rec[b].max())]
    ax.plot(lim, lim, "r--"); ax.set(xlabel="true", ylabel="recovered", title=f"{name}  (r = {np.corrcoef(rec[a], rec[b])[0,1]:.2f})")
fig.tight_layout(); fig.savefig(FIG / "mock_parameter_recovery.png", dpi=200)
""")
code(r"""
rows, curves_m = [], {}
for shape in ("falling", "flat", "rising"):
    m = make_mock_trials(20000, cbf_shape=shape, seed=1)
    b = bootstrap_cbf(m, n_boot=300, seed=2)
    rows.append({"true_shape": shape, "recovered_shape": b["shape"], "slope": b["slope"], "slope_lo": b["slope_lo"], "slope_hi": b["slope_hi"]})
    curves_m[f"mock: {shape}"] = (compute_cbf(m)["rt_median"].to_numpy(), b["cbf"])
shape_tab = pd.DataFrame(rows); shape_tab.to_csv(TAB / "mock_cbf_shape_recovery.csv", index=False)
print("CBF shape recovered for all 3 mock shapes:", bool((shape_tab.true_shape == shape_tab.recovered_shape).all()))
shape_tab.round(3)
""")
md(r"""
### 6.1 Cross-track check: the same tools on Track A's simulator
Qualitative reference only (default demo parameters from `run_track_a.py`, **not** fitted to the mice): the CBF tool should label the
M1 / M2 / M3 simulations falling / flat / rising. This is the contract that Week 3 relies on.
""")
code(r"""
cfg = {"M1 (starting point)": ("M1", {"a": 1.1, "t0": 0.2, "k": 2.6, "z": 0.40}),
       "M2 (drift bias)": ("M2", {"a": 1.1, "t0": 0.2, "k": 2.6, "v_b": 0.55}),
       "M3 (dynamic bias)": ("M3", {"a": 1.1, "t0": 0.2, "k": 2.6, "b_max": 1.2, "tau_b": 0.5})}
rows, ref = [], {}
for label, (mt, p) in cfg.items():
    sim = simulate_ddm(p, model_type=mt, n_trials=20000, seed=101)
    b = bootstrap_cbf(sim, n_boot=200, seed=0)
    rows.append({"model": label, "slope": b["slope"], "slope_lo": b["slope_lo"], "slope_hi": b["slope_hi"], "shape": b["shape"]})
    ref[label] = b["cbf"]
ref_tab = pd.DataFrame(rows); ref_tab.to_csv(TAB / "track_a_reference_cbf.csv", index=False)
plot_cbf_cohort(curves, overlay=ref, save_path=str(FIG / "cbf_cohort_with_reference_models.png"),
                title="Empirical CBF vs qualitative M1/M2/M3 reference (unfitted)");
ref_tab.round(3)
""")

md("## 7. Summary")
code(r"""
t = cohort_slope_test(slopes)
msg = f'''
**Data.** {len(df):,} trials from {df.subject_id.nunique()} mice (RT = response time); median RT {df.rt.median():.2f} s
(per-mouse medians {prof.rt_median.min():.2f}–{prof.rt_median.max():.2f} s).

**Psychometrics.** Prior shift positive in {int((ps['shift']>0).sum())}/{len(ps)} mice; {n_sig}/{len(ps)} have a session-level 95% interval excluding 0
(trial-level: {int((ps['shift_lo_trial']>0).sum())}/{len(ps)}). Mean shift {ps['shift'].mean():.3f} contrast units, mean ΔP(Right) at 0% contrast {ps['prior_effect_c0'].mean():.2f}.
Slope scale σ varies {prof.sigma.min():.3f}–{prof.sigma.max():.3f}; γ {prof.gamma.min():.2f}–{prof.gamma.max():.2f}.
The prior also changes the lapse asymptotes in some mice (γ right − left block: {prof.gamma_block_diff.min():+.2f} to {prof.gamma_block_diff.max():+.2f}).

**Chronometrics.** Mean RT falls from {mm.loc[0.0]:.2f} s (0% contrast) to {mm.loc[1.0]:.2f} s (100%).

**CBF.** Default selection (|contrast| ≤ 0.125, session-level bootstrap): cohort slope {t['mean_slope']:.3f} ± {t['sem_slope']:.3f} per RT bin (p = {t['t_p']:.3f}); per-mouse shapes {t['shape_counts']}.
By contrast selection: {"; ".join(f"{r.selection}: {r.mean_slope:+.3f} (p = {r.t_p:.3f})" for r in sens_c.itertuples())}.
By RT definition: {"; ".join(f"{r._1}: {r.mean_slope:+.3f} (p = {r.t_p:.3f})" for r in sens_rt.itertuples())}.
By block position (skip first n trials): {"; ".join(f"n={r._1}: {r.mean_slope:+.3f}" for r in sens_blk.itertuples())}.
Without KS021: slope {loo.mean_cbf_slope[1]:+.3f} (p = {loo.t_p[1]:.3f}).

**Heterogeneity.** Outliers: {"; ".join(f"{r.metric}: {r.outliers}" for r in hsum.itertuples() if r.outliers) or "none"}.

**Handoff to Week 3.** `results/empirical/cbf_bin_edges.csv` holds each mouse's RT quintile edges; pass them as `bin_edges` to
`compute_cbf` so model simulations are binned identically, and pass per-bin model CBFs to `plot_cbf_cohort(..., overlay=...)`.
'''
display(Markdown(msg))
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_track_b_empirical_baseline.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("written", len(cells), "cells")
