# INCM Project: Drift Diffusion Modeling of Mouse Perceptual Choice

**Course:** Introduction to Neural and Cognitive Modeling (INCM)  
**Authors:** Parth Dhodapkar (2024111009) & Joshua Koilpillai (2024101128)  
**Topic:** Evidence Accumulation (Question 27)  

---

## Overview

This repository investigates how an uncued, inferred block prior enters the decision process in head-fixed mice performing the International Brain Laboratory (IBL) 2AFC visual choice task. 

We compare three competing computational routes within the Drift Diffusion Model (DDM) framework:
1. **M1 (Starting Point Bias):** Prior shifts initial accumulator offset $x(0) = a \cdot z \cdot \text{prior}$ (falling Conditional Bias Function).
2. **M2 (Drift Bias):** Prior offsets the evidence accumulation rate $v(t) = v_{\text{sensory}} + v_h \cdot y_{t-1} + v_b \cdot \text{prior}$ (flat Conditional Bias Function).
3. **M3 (Dynamic Bias):** Urgency-dependent dynamic prior $v(t) = v_{\text{sensory}} + v_h \cdot y_{t-1} + \text{prior} \cdot b_{\max} \frac{t}{t + \tau_b}$ (rising Conditional Bias Function; Hanks et al., 2011).

Additionally, we evaluate **fixed vs. Weibull-collapsing boundaries (C2)** to test urgency effects on wheel turns and confirm invariance of the winning prior mechanism.

---

## Project Structure & Track Division

The project is structured according to the **Contract-First architecture** documented in [`plan.md`](plan.md):

```text
incm_project/
├── data/                    # IBL mouse datasets and cache
├── results/
│   ├── figures/             # Publication-grade figures (PNG/PDF)
│   └── recovery/            # Parameter and model recovery CSV benchmarks
├── src/
│   ├── models/              # [Track A - Parth] Simulator, boundaries (fixed & Weibull)
│   ├── optimization/        # [Track A - Parth] QMP loss, adaptive Nelder-Mead with CRN
│   ├── recovery/            # [Track A - Parth] Wilson & Collins (2019) recovery pipelines
│   ├── data/                # [Track B - Joshua] IBL ONE API loader and trial parser
│   ├── empirical/           # [Track B - Joshua] Empirical psychometrics, CBF, heterogeneity, mock data
│   └── visualization/       # Plotting routines (plots_recovery: Track A; plots_behavior: Track B; plots_c1: C1 figures)
├── notebooks/               # [Track B] 02_track_b_empirical_baseline.ipynb (Week 2 deliverable)
├── docs/DEVELOPER_GUIDE.md  # how everything is implemented, how to change it, where results live (start here when modifying code)
├── scripts/                 # Track A cluster job; build_empirical_notebook.py regenerates the notebook; C1/C2 checks and drivers
├── tests/
│   ├── test_data_contract.py# Schema validation & physical bounds
│   ├── test_simulator.py    # DDM physics & loss surface tests
│   ├── test_ibl_pipeline.py # [Track B] IBL preprocessing + processed-cohort contract tests
│   ├── test_empirical.py    # [Track B] mock-data harness for the empirical tools
│   ├── test_empirical_fit.py# Week 3: fitter extensions, shared cohort-fitting helpers
│   ├── test_c1.py           # Week 3: C1 tables, statistics and model-vs-data CBF
│   ├── test_c2.py           # Week 3: truncation-aware likelihood, warm starts, RT-tail tools, C2 tables and statistics
│   └── test_robust_c1.py    # Robust C1: inferred prior, lapse mixture, common-seed re-scoring, CBF-slope bootstrap
├── run_track_a.py           # CLI benchmark harness for Track A
├── run_track_b.py           # CLI for Track B: data pipeline and empirical notebook
├── run_c1.py                # Week 3 C1 runner: fit M1/M2/M3 to the mice, tables, statistics, overlays
├── run_c2.py                # Week 3 C2 runner: fixed vs Weibull bounds, invariance of C1, RT tail
├── plan.md                  # Detailed 2-person execution plan & interface contract
└── proposal.md              # Original project proposal
```

---

## Quickstart & Commands (Track A - Parth Dhodapkar)

### 1. Run Unit Test Suite
To verify the simulator physics, boundary monotonicity, dynamic bias, and data contract compliance:
```bash
pytest -v tests/
```

### 2. Run the End-to-End Pipeline Demo
Generates psychometrics, chronometrics, qualitative CBF curves for M1/M2/M3, and mini recovery runs:
```bash
python run_track_a.py --demo
```
Output figures are saved to `results/figures/`, and CSV benchmarks to `results/recovery/`.

### 3. Run Full Parameter Recovery Benchmark (Wilson & Collins Rule 4)
Runs 100 synthetic parameter sets across sample sizes ($N \in \{200, 500, 1000, 2000\}$) across all 16 CPU cores:
```bash
python run_track_a.py --full-param-recovery --model M2 --n-sims 100 --trials 200 500 1000 2000
```

### 4. Run Full Model Recovery Benchmark (Wilson & Collins Rule 5)
Simulates 50 synthetic datasets per model under M1, M2, and M3, fits all competing candidate models, and calculates the BIC confusion and inversion matrices:
```bash
python run_track_a.py --full-model-recovery --n-datasets 50
```

### 5. Plan-scale recovery results (Week 2 deliverable, run in Week 3)
`bash scripts/local_recovery.sh 13` runs everything above for M1, M2, M3 and JOINT on one workstation (same jobs as `scripts/ada_recovery.sbatch`;
~80 CPU-hours, about 9 h wall-clock on a 20-thread / 7 GB laptop with 13 workers, checkpointed and resumable; every finished fit is stored in
`results/recovery/*.partial.csv`). `python scripts/summarize_recovery.py` writes `results/recovery/recovery_report.md` (matrices, r / slope tables, error correlations);
CSVs and figures are in `results/recovery/`. Results against the plan's criteria:
- **Model recovery (150 datasets, N = 1000): criterion NOT met.** P(fit | true) = M1 94%, M2 90%, **M3 68%** (overall 84%; criterion > 85% for every mechanism).
  All 16 M3 failures were won by M2, by 5.5 BIC on average (about the one-parameter BIC penalty at N = 1000), and they sit in weakly biased draws: recovery is 30-40% for b_max <= 0.8 but 87-93% above it.
  So M3 is only identifiable when its bias is substantial, and at this size a weak M3 is indistinguishable from M2. (The real-data design with 6,000+ trials is easier: `results/recovery/empirical_model_recovery.csv`, 18/18.)
- **Parameter recovery (100 vectors x N = 200/500/1000/2000):** every M1 and M2 parameter has r > 0.8 at every N (z: r >= 0.98, v_b: r >= 0.98), slopes 0.9-1.05.
  JOINT passes except k at N = 200 (r = 0.78). M3: b_max reaches r = 0.82-0.87 for N >= 500 (0.78 at 200) but **tau_b is not recoverable (r = 0.12-0.32)**, which is why M3 profiles tau_b over a grid and counts it in BIC.
- **Cross-talk:** z and v_b do not trade off when fitted separately, but **in the JOINT model their errors are strongly negatively correlated (-0.57 to -0.67 at every N)**, so z and v_b cannot be separated when both are free;
  other trade-offs: a-k (M1, up to 0.57), a-t0 (M2, about -0.5), b_max-tau_b (M3, up to 0.80).

---

## Quickstart & Commands (Track B - Joshua Koilpillai)

### 0. Environment
`ONE-api` is all that is needed to read IBL trial tables (`ibllib` is not required):
```bash
python3 -m venv ~/.venvs/incm && source ~/.venvs/incm/bin/activate
pip install -r requirements.txt
```

### 1. Build the cleaned 10-mouse IBL dataset
Connects to the public IBL Alyx server (no registration; shared public credentials), downloads
the trial tables of all standard `biasedChoiceWorld` training sessions into `data/raw/ONE/`
(git-ignored, ~170 MB), selects 10 mice and writes `data/processed/`. Small copies of the QC tables
(`session_qc.csv`, `candidate_sessions.csv`) are kept in `data/processed/` so a fresh clone can re-select without downloading:
```bash
python run_track_b.py --all          # ~40 min the first time (~2,700 small files); resumable
python run_track_b.py --process      # rebuild processed files from the cached downloads only (~7 min)
python run_track_b.py --refresh-qc   # recompute session QC from the local cache (offline)
python run_track_b.py --test         # pytest suite (includes the data-contract tests)
```

Selection rule (deterministic): standard IBL biased training sessions (project
`ibl_neuropixel_brainwide_01`, trials table available); a session qualifies with >= 400 trials,
both block types, and >= 80% accuracy on easy trials (|contrast| >= 0.25); a mouse qualifies with
>= 5 qualifying sessions; mice are ranked by total valid trials with at most 2 per lab.
**The cohort is frozen once chosen:** `--select` / `--process` keep the mice stored in
`data/processed/mice_selection.csv` and only refresh the ranking columns (pass `--reselect` to adopt a fresh ranking).
Ranking by valid trials with the response-time `rt` below would pick 8 different mice (the old ranking favoured mice whose
first-movement times were slow), so the original 10 were kept.

Outputs in `data/processed/` (all conform to the §3.1 schema):
- `cohort_rt2p5.{parquet,csv}` / `cohort_rt5p0.{parquet,csv}` — all 10 mice (RT in [0.1, 2.5] s for C1; [0.1, 5.0] s for C2)
- `rt_2p5/<mouse>.{parquet,csv}`, `rt_5p0/<mouse>.{parquet,csv}` — one file per mouse
- `cohort_extended.parquet` — 5.0 s cohort plus session id/date, lab, `block_pos` (trials since the last block change),
  `rt_response`, `rt_firstmove`, feedback and `rt_in_2p5` flag
- `summary_stats.csv`, `mice_selection.csv`, `session_manifest.csv` — statistics, mouse ranking, included sessions

Conventions: `prior = +1` is a right-biased block (P(left stimulus) = 0.2), `choice = +1` is a
Right response, `rt = response_times - stimOn` (stimulus onset to wheel-threshold crossing), `prev_choice` is
the true 1-back choice (0 at session start and after a no-go). The unbiased opening trials of each session
(P(left) = 0.5) are dropped. The first-movement time `firstMovement_times - stimOn` is kept as `rt_firstmove`
(`--rt-source first_movement` makes it the contract `rt`): it is <= 0.1 s on ~45% of responded trials
(anticipatory wheel movement), and those trials are not a random subset (the choice follows the block more often),
so using it with the 0.1 s filter discarded half the data and biased the CBF.

### 2. Empirical baseline (Week 2): psychometrics, chronometrics, CBF, heterogeneity
```bash
python run_track_b.py --empirical     # executes notebooks/02_track_b_empirical_baseline.ipynb (a few minutes; any working directory)
python scripts/build_empirical_notebook.py   # regenerates the notebook file from its source
pytest tests/test_empirical.py        # mock-data harness (known ground truth) + schema-generality tests
```
The notebook writes tables to `results/empirical/` and figures to `results/figures/empirical/`.

For a full description of the code (and recipes for changing it) see [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md).

`src/empirical/` (all functions take any §3.1 DataFrame — IBL data, simulator output or mock data):
- `psychometrics.py` — P(Right) tables with Wilson CIs, lapse-corrected logistic fits (γ = P(Right) for strongly left stimuli, λ = P(Left) for strongly right stimuli), block-prior shift with per-block lapses, chronometric / RT-quantile / conditional-accuracy tables
- `cbf.py` — **contrast-stratified** Conditional Bias Function over RT quintiles, trial- or session-level (`cluster_col="session_eid"`) bootstrap CIs, slope and shape label (falling / flat / rising)
- `heterogeneity.py` — one-row-per-mouse behavioural profile (incl. the prior's effect on each lapse and a within-block history effect), between-mouse summary and correlations
- `tails.py` — model-free RT-tail diagnostics for C2 (survival, hazard, late/early hazard ratio, accuracy by RT window)
- `mock.py` — synthetic §3.1 generator with known psychometric parameters, prior strength and falling / flat / rising bias, independent of the Track A simulator

For Week 3 (Track A overlay): `cbf_curve(df)` is a drop-in for `compute_cbf_curve`; `compute_cbf(sim_df, bin_edges=...)`
accepts each mouse's RT bin edges from `results/empirical/cbf_bin_edges.csv`; `plot_cbf_cohort(curves, overlay={...})` overlays model CBFs.
`src/visualization/plots_behavior.py` holds the plotting functions.

### 3. Week 3: fitting the DDM family to the mice (C1 and C2)
`src/optimization/empirical.py` is the shared entry point, so C1 (M1/M2/M3, fixed bounds) and C2
(fixed vs Weibull, 2.5 s vs 5.0 s) use identical settings:
```python
from src.optimization.empirical import fit_cohort, simulate_from_fit, delta_bic, fitted_params
c25 = pd.read_parquet("data/processed/cohort_rt2p5.parquet")
res = fit_cohort(c25, models=("M1", "M2", "M3"), bounds=("fixed",), rt_max=2.5, cache_path="results/fits/c1_fixed_2p5.csv")
c50 = pd.read_parquet("data/processed/cohort_rt5p0.parquet")        # C2 extended cutoff: rt_max=5.0
```
Fits run in parallel and resume from `cache_path` (use a different cache file per configuration).
Why the empirical settings differ from `fit_ddm`'s defaults (all found by piloting on the real mice):
- `fit_beta=True` — with the contrast exponent fixed at 1 the model under-predicts 6%-contrast accuracy by ~25 points; the fitted exponent is small (~0.15-0.35).
- `fit_variability=True` — `s_v`, `s_z` (plan §4); used in BOTH fixed and Weibull fits so C2 compares like with like.
- `bias_type='drift'` — a neutral side bias `v_0` in every model; 7/10 mice favour one side by 10-25 points at 0% contrast, which `z` / `v_b` / `b_max` would otherwise absorb.
- `t_max` = the RT cutoff (the default 2.5 s cannot simulate the 5.0 s data set).
- Heavy search (`n_screen=80, n_starts=5, n_restarts=3`) — with the light defaults the BIC of the same fit varied by ~660 across seeds on one mouse; the heavy search is better (about 8 min of CPU per fit for M1/M2, ~3x for M3). **Single fits are still noisy: the C1 replicate fits on the full cohort gave a pooled run-to-run SD of ~250 BIC (per-fit spread 3-1041), so always run replicate seeds (`fit_cohort(seeds=...)`) and do not read per-mouse differences below a few hundred BIC.**

Known limits of the model class (report them, they bound what C1/C2 can say): there is no lapse process, and some
mice (KS019, KS021, CSH_ZAD_002, NYU-01) make *block-dependent* lapse errors at high contrast, which no drift/start-point
mechanism reproduces; their fits are mis-specified and should be shown with and without them. `delta_bic(...)` builds
per-mouse BIC differences; `simulate_from_fit(...)` replays a mouse's covariates through a fit for the model-vs-data CBF overlay.
`python scripts/validate_empirical_recovery.py` checks M1/M2/M3 recovery for this extended model class on a real trial design.
Result (6 datasets per model, 6,000 trials, NYU-09's trial design; `results/recovery/empirical_model_recovery.csv`): 18/18 datasets
won by the true model, but 4/18 by less than 10 BIC, all in the M2/M3 pair — so M2 vs M3 is the weakest contrast in Week 3.

### 4. Week 3, C1: which route does the prior take? (`run_c1.py`)
```bash
python run_c1.py fit                          # 10 mice x M1/M2/M3 x 2 seeds = 60 heavy fits, 12 workers: ~2 h awake (resumable)
python run_c1.py fit --variant no_history --n-jobs 6   # sensitivity without the choice-history drift v_h: ~15 min
python run_c1.py fit --variant wide_bounds --n-jobs 5  # sensitivity with relaxed t0 / s_z search bounds: 60 fits, ~5 h on 5 workers while sharing the CPU
python run_c1.py fit --variant joint --n-jobs 5        # follow-up: combined z + v_b model, 20 fits: ~1.5 h on 5 workers
python run_c1.py analyze                      # tables, tests, posterior-predictive CBFs, figures: ~30 s
python run_c1.py smoke                        # tiny end-to-end check (3 mice, seconds)
python scripts/check_m1_start_point.py        # M1 with z forced to 0.15/0.3/0.45 on two mice: ~12 min
python scripts/check_param_bounds.py          # are fits limited by the t0 / s_z bounds? (2 mice x seeds, ~15 min)
```
Outputs: `results/c1/c1_report.md` (all tables), `c1_bic_table.csv` (per-mouse BIC and ΔBIC vs M2), `c1_pairwise_tests.csv`
(paired Wilcoxon / t / sign tests, bootstrap CI over mice, Holm-adjusted), `c1_subset_tests.csv` (without the lapse-affected mice
or KS021), `c1_cbf_fit.csv` / `c1_cbf_model_tests.csv` (predicted vs empirical CBF), `c1_concordance.csv`, `c1_parameters.csv`,
`c1_params_at_bounds.csv`; figures in `results/figures/c1/`. Code: `src/optimization/c1.py` (analysis), `src/visualization/plots_c1.py`.
Conventions: ΔBIC = BIC(model) − BIC(reference), negative = better. Each (mouse, model) is fitted from two independent seeds; the
best-BIC run is kept and the replicate spread gives the noise floor (`decisive` = ΔBIC above max(10, 2.8 × pooled SD) = 708 BIC).
The predicted CBF replays each mouse's own trials 20 times through its fitted model and is binned on that mouse's empirical RT bins.

**Result (10 mice, 2.5 s cutoff, fixed bounds; all three models have β, s_v, s_z, v_h and a neutral side bias):**
- **M1 (start point) is rejected.** M1 is worse than M2 in 10/10 mice and worse than M3 in 10/10: mean ΔBIC +369 vs M2 (bootstrap 95% CI
  165 to 597; Wilcoxon p = 0.002, Holm 0.006) and +368 vs M3. Its fitted start-point bias collapses to |z| ≤ 0.1 in all mice; forcing z = 0.15-0.45 makes the fit far worse
  (ΔNLL of +190 to +600 at 0.15 and thousands at 0.45 on NYU-09 / ibl_witten_13), so this is not an optimizer failure. Dropping the
  history term v_h leaves it unchanged (M1 worse in 9/10, mean +1,140, Holm p = 0.008). Predicted M1 CBFs are far below the data (0.1 vs 0.25-0.35).
- **M2 (drift bias) vs M3 (dynamic bias) is not separable by BIC**: mean ΔBIC(M3 − M2) = +1 (CI −126 to 97; Wilcoxon p = 0.63); M2 has the lowest BIC in 6 mice, M3 in 4,
  and no single mouse exceeds the noise threshold. (The recovery study already flagged this pair as the weakest contrast.)
- **The CBF shape favours M2 over M3.** Empirical CBFs are falling (4 mice) or flat (6), none rising. M3 fits predict a rising CBF in 9/10 mice (+0.01 to +0.04 per bin);
  M2 predicts the mild decline that a drift bias produces (−0.01 to −0.04) which matches. M2's CBF is closer to the data than M3's in 8/10 mice
  (RMSE +0.026 for M3, Wilcoxon p = 0.02, Holm-adjusted) and than M1's in 10/10 (p = 0.006).
- **Verdict:** the prior enters as a drift/choice-bias (M2) rather than a start-point shift (M1); M3 is not excluded by likelihood but is contradicted by the CBF shape.

Robustness and caveats (report these alongside the result):
- *Mis-specified mice.* Without KS019, KS021, CSH_ZAD_002 and NYU-01 (block-dependent lapses) the M1-vs-M2 effect has the same size (+372, CI 55 to 702) but with n = 6 the Wilcoxon test cannot go below p = 0.031 (Holm 0.09). Three of the four M3 winners are among these mice.
- *All models under-predict the fastest RT bin* (cohort CBF in bin 1: data 0.347, M2 0.27, M3 0.17, M1 0.18). A combined start-point + drift model does not fix it:
  JOINT (`run_c1.py fit --variant joint`) fits z = -0.14 to 0.02 in all 10 mice, has a worse BIC than M2 in 9/10 (mean +160) and a CBF no closer to the data (`c1_joint_*`).
  Caveat: z and v_b trade off inside JOINT (error correlation about -0.6 in the recovery study), so this is evidence against an added start-point route, not a clean estimate of its size.
- *Search bounds are binding.* With the default bounds (t0 >= 0.08 s, s_z <= 0.6) 5/10 mice sit on t0 and 4-5/10 on s_z. Relaxing them (t0 >= 0.02, s_z <= 1.2) typically improves BIC by 60-770 per fit (a few fits come out worse, which is search noise), and then
  s_z rises above 1.0 in most mice and reaches the NEW ceiling of 1.2 in 6/10 mice in at least one model (1/10 in all three; `c1_wide_bounds_parameters.csv`): a start-point spread wider than the bounds produces fast responses at t0 = 0.10 s, i.e. the data want a **fast-guess component** that this model family lacks (like the lapses, a model-class limit; tested directly by the `fastguess` variant, section 6).
  The full refit with relaxed bounds (`c1_wide_bounds_*`, `results/figures/c1/c1_cbf_overlay_cohort_wide_bounds.png`) leaves the conclusions intact: M1 worse than M2 in 9/10 mice (mean ΔBIC +512, CI 227 to 836, Wilcoxon p = 0.004, Holm 0.012);
  M3 vs M2 still null (mean -5, CI -181 to 124, p = 0.43); M2's CBF is closest to the data in 8/10 mice.
- *Optimizer noise* is large (pooled SD ≈ 250 BIC) and was only reduced by two seeds; the group-level results are the reliable ones, per-mouse winners are not.
- *Choice history.* Because the previous choice follows the block, v_h partly carries block information in every model; the no-history refit (`c1_no_history_*`) shows the ranking does not depend on it.
- `SE` in the CBF discrepancy (`chi2_per_bin`) assumes independent trials and is descriptive only.


### 5. Week 3, C2: is there urgency (collapsing bounds), and does C1 survive it? (`run_c2.py`)
```bash
python run_c2.py fit                              # fixed + Weibull fits, 3 variants, warm starts + polish: ~8 h on 12 workers (resumable)
python run_c2.py fit --variant tail_5p0           # sensitivity with a tail-weighted likelihood: ~1.7 h
python scripts/validate_c2_recovery.py            # can the pipeline recover the true bound type? ~1.7 h
python scripts/profile_collapse.py                # likelihood profile over the collapse latency: ~12 min
python run_c2.py analyze                          # all tables, tests, tail analysis, figures: ~2.5 min
python run_c2.py smoke                            # tiny end-to-end check (seconds)
bash scripts/run_c2_all.sh 12                     # everything in order
```
Outputs: `results/c2/c2_report.md` (all tables), `results/c2/c2_*` (per-variant tables), `results/c2/validation/` (recovery), figures in `results/figures/c2/`.
Code: `src/optimization/c2.py`, `src/empirical/tails.py`, `src/visualization/plots_c2.py`; design details, variants and how to change them: `docs/DEVELOPER_GUIDE.md` section 9.

What was done differently from a naive "refit with a Weibull bound" (each point found by checking, details in the guide):
- **Nesting.** Track A's default `lam` range (0.4-3.5) cannot reproduce a fixed bound; C2 fits use `lam` up to 20 so that fixed is the limit of Weibull, and every Weibull fit is warm-started from the fixed solution.
- **Truncation-aware likelihood** (`fit_mouse(truncate=True)`, opt-in): 6-23% of responses are slower than 2.5 s and were removed, but the C1 likelihood still lumps simulated slow mass into the last bin (the fixed-bound fits predicted only 0.1-3% of RTs above 2.5 s), which biases towards collapse. The C2 model is scored conditional on the RT window.
- **Polish pass** and replicate seeds, because the validation showed fixed and Weibull fits ending in optima up to 50 NLL apart in either direction.
- **Validation on simulated data**: a true collapse is found in 12/12 datasets, but at the 2.5 s cutoff a fixed bound is called collapsing in 25-75% of datasets (also with disengaged-trial contamination); at 5.0 s no fixed-bound dataset was called collapsing by more than 10 BIC. Per-mouse 2.5 s collapse claims are therefore not reliable.

**Result (10 mice, model M2; ΔBIC = Weibull − fixed, negative = collapsing better; no single mouse exceeds the noise threshold of 610-1270 BIC, so statements are group-level):**
- **No evidence for urgency.** At the 5.0 s cutoff all 10 mice fit the fixed-bound limit (`lam` at its ceiling) and the fixed bound is preferred in 9/10 (mean ΔBIC +12 [3.5, 18], about the 2-parameter penalty; Wilcoxon p = 0.049). At 2.5 s the group mean is −10.6 (p = 1.0; 3 mice look collapsing, but see the validation) and with the C1 likelihood +14.4 (p = 0.08).
  A tail-weighted likelihood (extra quantiles 0.95/0.99) lets 3 mice (CSHL046, CSH_ZAD_002, NYU-09) prefer a late, mild collapse (`lam` ≈ 5 s) but the group mean (−94 [−224, 19], p = 0.63) is not significant, three other mice show convergence failures, and the tail fit barely improves.
  The profile likelihood rejects any collapse beginning before ~3 s (ΔNLL +650 to +12,000).
- **The C1 conclusion is invariant to the bound.** Under Weibull bounds M1 is still worse than M2 (+322 BIC vs +318 under fixed; shift +3.6 BIC [−8, 21], ≈ 1% of the effect), M2 vs M3 is still not separable (−23 vs −22), per-mouse contrasts correlate at ρ = 0.93-0.96, and the model-predicted CBF is still closest to the data for M2. This is close to trivial because the Weibull fits sit at the fixed-bound limit.
- **The RT tail argues against urgency.** The mice have a heavier slow tail than the fitted model (median 2.9× the predicted fraction of RTs > 2.5 s; 10/10 mice), a more strongly rising late hazard (10/10) and *higher* accuracy in slow responses (+17 points, 9/10), whereas urgency would lighten the tail and push late accuracy to chance. The Weibull model reproduces the tail no better than the fixed one. The extra slow, accurate responses suggest a slow/disengaged component the model class lacks (also relevant to the lapse-affected mice).
- Caveats: optimizer noise (pooled SD 220-450 BIC per fit); 2.5 s per-mouse results unreliable; one lam-profile on two mice only; model class has no lapse/fast-guess/disengagement process.

### 6. Robust C1 (v2): fixing the weak points of the Week 3 result
The Week 3 C1 result has three weak points: the fits are noisy (pooled SD ≈ 250 BIC, so no mouse is decisive), the model class has no
lapse / fast-guess process (four mice are mis-specified and `s_z` runs into its ceiling), and the prior covariate is the TRUE block label,
which the mouse never sees. Robust C1 adds one fix per variant, so their effects can be told apart:

| variant | adds | fits |
|---|---|---|
| `wide_bounds` (existing) | t0 ≥ 0.02 s, s_z ≤ 1.2 | 10 mice × 3 models × 2 seeds |
| `lapse` | + lapse / contaminant mixture: rate `lapse`, P(Right \| lapse) = sigmoid(`lapse_side` + `lapse_prior` · prior), RT uniform on [0.1, 2.5] s (3 parameters, same for every model) | × 3 seeds |
| `robust` | + **inferred prior**: the Bayes-optimal observer's belief about the block (`src/data/inferred_prior.py`), binned into 4 levels for the likelihood | × 3 seeds |

Both new variants are then **re-scored** (`run_c1.py rescore`): every replicate of every model is evaluated once more with 4,000
simulations per condition and the SAME seed, so models are compared on common random numbers. `analyze` adds a **calibrated CBF-slope
test** for M2 vs M3 (a parametric bootstrap: 200 synthetic experiments of the mouse's own size per fitted model) and a model-free check of
which prior covariate the choices follow. BIC is only comparable across models *within* a variant (the inferred-prior variant has more
likelihood conditions).

Already known without any new fits (`results/c1/c1_prior_covariate_comparison.csv` after `analyze`): in a logistic regression with
the same number of parameters, **the inferred prior explains the choices better than the true block label in 10/10 mice** (log-likelihood
+21 to +362). The mice track the block the way an ideal observer would, lagging after each switch, so the true label mis-states their prior
on exactly the trials that carry information about it.

Cost on one 38-core Ada node (CPU only): `robust` 2 h 10 min, `lapse` 1 h 15 min, each including re-scoring and analysis.

**Result (round 2; `results/c1/c1_report.md`, sections "Robust C1"):**
- Run-to-run noise fell from SD ≈ 250 to 86 BIC (`robust`), so a per-mouse difference is decisive above 242 instead of 708.
- **M1 is still rejected**: worse than M2 in 9/10 mice, mean ΔBIC +320 (CI 114 to 550), Wilcoxon Holm p = 0.018 (`lapse`: +237, Holm p = 0.012).
- **M2 vs M3 is still not separable by BIC** (mean +5, p = 0.92; 4 vs 5 winners; KS019 decisively M3, ibl_witten_13 decisively M2).
- **The calibrated CBF-slope test favours M2 over M3 in 10/10 mice** (summed log LR 363, Wilcoxon p = 0.002); M3's rising CBF is rejected in every mouse.
  But the observed CBF falls more steeply than M2 predicts in 6/10 mice: M2 is far less wrong than M3, not a complete account (fast-guess test below).
- Lapses are 9–11% of trials and are biased towards the block's side in every mouse (`lapse_prior` > 0 in all 30 `lapse` fits): the prior also enters guesses.

#### Running it on Ada
Copy the working tree from the laptop (Git Bash, from the folder that contains `incm_project/`; no commit needed;
IIIT VPN required off campus; check that `incm_project.tgz` arrives at its full size, about 48 MB):
```bash
tar --exclude=incm_project/.git --exclude=incm_project/data/raw --exclude=__pycache__ --exclude=.pytest_cache -czf incm_project.tgz incm_project
scp incm_project.tgz <user>@ada-gw1.iiit.ac.in:
ssh <user>@ada-gw1.iiit.ac.in 'tar xzf incm_project.tgz && rm incm_project.tgz'
```
One-time Python setup. Build the environment ON A COMPUTE NODE: the gateway runs a different OS (system Python 3.9)
from the u22 compute nodes, and an environment built on the gateway fails in jobs with `No module named 'numpy'`.
```bash
ssh <user>@ada-gw1.iiit.ac.in
srun -A research --qos=medium -c 4 --mem-per-cpu=2G --time=01:00:00 --pty bash -l   # interactive compute node
module load u22/python/3.12.4 && python3 --version                                    # must say 3.12.x
rm -rf ~/incm_env && python3 -m venv ~/incm_env && source ~/incm_env/bin/activate
pip install -r ~/incm_project/requirements.txt
cd ~/incm_project && python -m pytest -q tests/test_robust_c1.py && exit
```
Submit (from the gateway). Ada caps a job at 38 CPUs, so ask for 38:
```bash
cd ~/incm_project && mkdir -p logs
JOB=$(sbatch --parsable -c 38 --mem-per-cpu=1500M --export=ALL,VARIANT=robust scripts/ada_c1_robust.sbatch)
sbatch -c 38 --mem-per-cpu=1500M --dependency=afterok:$JOB --export=ALL,VARIANT=lapse scripts/ada_c1_robust.sbatch   # ablation, after robust succeeds
```
Watching a run:
```bash
squeue -u $USER                       # state (PD pending / R running), elapsed time, node; the NODELIST(REASON) column says why it waits
tail -f logs/c1_<id>.out              # live log: "[Parallel(n_jobs=40)]: Done N tasks | elapsed" lines, then "fits done", "rescore done", "ALL DONE"
python scripts/c1_progress.py         # finished fits per variant, e.g. "robust  fits  37/90  [M1 12, M2 13, M3 12]  rescored 0/90"
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode    # after it ends: final state, run time, peak memory
scancel <id>                          # stop it; resubmitting the same command resumes from the saved fits
```
If the job ends with `TIMEOUT` or is killed, submit it again: finished fits are in `results/c1/c1_fits_<variant>.csv` and are skipped.
These two jobs are CPU-only; round 3 below runs the simulator on the node's GPUs.

Fetch the results to the laptop (Git Bash, from `incm_project/`):
```bash
scp -r <user>@ada-gw1.iiit.ac.in:incm_project/results/c1 results/
scp -r <user>@ada-gw1.iiit.ac.in:incm_project/results/figures/c1 results/figures/
```
Outputs: the `## Robust C1 (...)` sections and the prior-covariate table in `results/c1/c1_report.md`; per variant `c1_<variant>_bic_table.csv`,
`_pairwise_tests.csv`, `_subset_tests.csv`, `_cbf_fit.csv`, `_cbf_slope_bootstrap.csv` / `_cbf_slope_lr.csv` (M2 vs M3), `_parameters.csv` (incl. lapse);
figures `results/figures/c1/c1_<variant>_delta_bic.png`, `c1_cbf_overlay_cohort_<variant>.png`, `c1_cbf_overlay_by_mouse_<variant>.png`.

#### Round 3: validation, fast guesses, replication, fairer M3 (GPU)
| step | what it answers | code / output |
|---|---|---|
| model recovery | If a mouse really were M1 / M2 / M3 with the parameters we fitted, would the robust pipeline say so? (each mouse's own design, 90 fits) | `scripts/validate_robust_recovery.py` → `results/recovery/robust_model_recovery*.{csv,md}` |
| `fastguess` | Do prior-biased anticipatory responses (RT uniform just above 0.1 s, choice biased by the prior; 3 parameters) explain the steeply falling CBF that M2 misses? | `run_c1.py --variant fastguess` → `c1_fastguess_*` |
| replication | Do the conclusions hold in 10 NEW mice (next 10 in the same ranking, ≤ 2 new per lab)? | `run_track_b.py --replication` → `data/processed/replication/`; `results/c1_replication/` |
| `robust_tau` | Was M3 handicapped by its 3-value tau_b grid? M3 refit with 7 values (a superset), merged with the robust fits as `robust_finetau` | `run_c1.py --variant robust_tau` → `c1_robust_finetau_*` |

**GPU simulator.** `INCM_BACKEND=cuda` runs the Euler–Maruyama kernel on an NVIDIA GPU (`src/models/_kernel_cuda.py`, numba-cuda,
float32, same random streams, so common random numbers still hold). One simulation of 33,600 trials: 7–17 ms on an RTX 4050 vs 230 ms on
one CPU core; a pool of fit workers sharing the GPU ran 3× faster end to end with identical BIC. The CPU still computes the likelihood,
so expect roughly a 2.5–3× speed-up per node, not 30×. Without a usable GPU the code falls back to the CPU.

One-time GPU setup on Ada (on a GPU compute node, inside the existing environment):
```bash
srun -A research --qos=medium -c 4 --gres=gpu:1 --mem-per-cpu=2G --time=00:30:00 --pty bash -l
module load u22/python/3.12.4 && source ~/incm_env/bin/activate && cd ~/incm_project
nvidia-smi                                   # driver >= 525 for CUDA 12; otherwise install numba-cuda[cu11]
pip install -r requirements-gpu.txt
python -m pytest -q tests/test_robust_c1.py  # includes the GPU-vs-CPU test (skipped without a GPU)
exit
```
Then, from the gateway: `mkdir -p logs && sbatch scripts/ada_round3.sbatch`. Each step resumes from saved fits; the log marks
`=== STEP`, `=== DONE` / `=== FAILED` and ends with `ALL DONE` or the list of failed steps. Progress:
`python scripts/c1_progress.py fastguess robust_tau` and `python scripts/c1_progress.py robust --out results/c1_replication`
(both need the environment, so run them in an `srun` session), or `wc -l results/c1/c1_fits_fastguess.csv` on the gateway.
Fetch: `results/c1`, `results/c1_replication`, `results/recovery`, `results/figures/c1`, `results/figures/c1_replication`.

---

## Interface Contract (§3 in `plan.md`)

### Trial DataFrame Schema
All simulation and empirical data loaders output a `pandas.DataFrame` with:
- `subject_id` (`str`): Subject or simulation identifier
- `trial_idx` (`int`): Trial index ($0$ to $N-1$)
- `contrast` (`float`): Signed visual contrast ($-1.0$ to $1.0$)
- `prior` (`float`): Block prior ($-1.0$ for left 20/80, $+1.0$ for right 80/20)
- `prev_choice` (`int`): Previous choice ($-1$ Left, $+1$ Right, $0$ initial)
- `choice` (`int`): Current choice ($-1$ Left, $+1$ Right)
- `rt` (`float`): Reaction time in seconds ($\ge t_0$)
