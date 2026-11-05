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
│   └── visualization/       # Plotting routines (plots_recovery: Track A; plots_behavior: Track B)
├── notebooks/               # [Track B] 02_track_b_empirical_baseline.ipynb (Week 2 deliverable)
├── scripts/                 # Track A cluster job; build_empirical_notebook.py regenerates the notebook
├── tests/
│   ├── test_data_contract.py# Schema validation & physical bounds
│   ├── test_simulator.py    # DDM physics & loss surface tests
│   ├── test_ibl_pipeline.py # [Track B] IBL preprocessing + processed-cohort contract tests
│   └── test_empirical.py    # [Track B] mock-data harness for the empirical tools
├── run_track_a.py           # CLI benchmark harness for Track A
├── run_track_b.py           # CLI for Track B: data pipeline and empirical notebook
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

`src/empirical/` (all functions take any §3.1 DataFrame — IBL data, simulator output or mock data):
- `psychometrics.py` — P(Right) tables with Wilson CIs, lapse-corrected logistic fits (γ = P(Right) for strongly left stimuli, λ = P(Left) for strongly right stimuli), block-prior shift with per-block lapses, chronometric / RT-quantile / conditional-accuracy tables
- `cbf.py` — **contrast-stratified** Conditional Bias Function over RT quintiles, trial- or session-level (`cluster_col="session_eid"`) bootstrap CIs, slope and shape label (falling / flat / rising)
- `heterogeneity.py` — one-row-per-mouse behavioural profile (incl. the prior's effect on each lapse and a within-block history effect), between-mouse summary and correlations
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
- Heavy search (`n_screen=80, n_starts=5, n_restarts=3`) — with the light defaults the BIC of the same fit varied by ~660 across seeds; the heavy search brings it to ~60 (about 8 min per 22k-trial fit). **Treat |ΔBIC| < ~100 as inconclusive.**

Known limits of the model class (report them, they bound what C1/C2 can say): there is no lapse process, and some
mice (KS019, KS021, CSH_ZAD_002, NYU-01) make *block-dependent* lapse errors at high contrast, which no drift/start-point
mechanism reproduces; their fits are mis-specified and should be shown with and without them. `delta_bic(...)` builds
per-mouse BIC differences; `simulate_from_fit(...)` replays a mouse's covariates through a fit for the model-vs-data CBF overlay.
`python scripts/validate_empirical_recovery.py` checks M1/M2/M3 recovery for this extended model class on a real trial design.
Result (6 datasets per model, 6,000 trials, NYU-09's trial design; `results/recovery/empirical_model_recovery.csv`): 18/18 datasets
won by the true model, but 4/18 by less than 10 BIC, all in the M2/M3 pair — so M2 vs M3 is the weakest contrast in Week 3.

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
