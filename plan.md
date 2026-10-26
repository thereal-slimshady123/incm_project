# INCM Project Execution Plan: Fully Decoupled 2-Person Parallel Roadmap

**Project:** How Does an Uncued, Inferred Prior Enter the Decision Process? A Drift Diffusion Model Study of Mouse Perceptual Choice  
**Course:** Introduction to Neural and Cognitive Modeling (INCM)  
**Team Members:** Parth Dhodapkar (2024111009) & Joshua Koilpillai (2024101128)  

---

## 1. Parallelization Philosophy: The "Contract-First" Architecture

The biggest danger in 2-person computational neuroscience projects is **serial dependency**: Person B waiting for Person A to write the simulation engine before starting data fitting, or Person A waiting for Person B to download and clean empirical data before testing the pipeline.

To achieve **100% parallel execution from Day 1**, we divide the project into two distinct, decoupled tracks:

- **Track A (Computational Engine & Synthetic Identifiability):** Focuses strictly on theoretical validity, simulation physics, optimization algorithms, and the **Wilson & Collins (2019)** parameter/model recovery pipeline on synthetic ground truth. **Requires 0 empirical data to start and complete.**
- **Track B (Empirical Data & Biological Adjudication):** Focuses on IBL mouse behavioral data ingestion, trial filtering, empirical behavioral characterization, running the empirical fits (C1 & C2), and biological interpretation. **Requires 0 simulation engine code on Day 1 (uses synthetic mock tables and standard baselines until integration).**

```
                  DAY 1: JOINT CONTRACT AGREEMENT
            (Standardized Trial DataFrame & Model API)
                           /        \
                          /          \
   TRACK A: COMPUTATIONAL ENGINE    TRACK B: EMPIRICAL DATA & BEHAVIOR
   (Parth Dhodapkar)                (Joshua Koilpillai)
   - Vectorized Euler–Maruyama      - IBL ONE API Ingestion (10 mice)
   - QMP Objective & Nelder–Mead    - Behavioral Preprocessing & Cleaning
   - Parameter Recovery (400 fits)  - Empirical Psychometrics & Chronometrics
   - Model Recovery (450 fits)      - Empirical CBF Curves per Mouse
   - Confusion/Inversion Matrices   - Mock Data Generation for Testing
                          \          /
                           \        /
                 WEEK 3: SEAMLESS INTEGRATION
         - Fit Validated Engine to Cleaned IBL Data
         - Adjudicate C1 (M1 vs M2 vs M3 via ΔBIC & CBF)
         - Adjudicate C2 (Fixed vs Collapsing Bounds)
         - Posterior Predictive Checks & Final Publication Figures
```

---

## 2. Core Ownership & Role Division

| Domain | Parth Dhodapkar (Track A) | Joshua Koilpillai (Track B) |
|---|---|---|
| **Primary Focus** | Computational Modeling & Identifiability | Rodent Neurobiology & Empirical Data |
| **Key Framework** | Wilson & Collins (2019) 10 Simple Rules | International Brain Laboratory (IBL) Pipeline |
| **Code Modules** | `src/models/`, `src/optimization/`, `src/recovery/` | `src/data/`, `src/empirical/`, `src/visualization/` |
| **Deliverables** | DDM simulator, QMP fitter, Recovery Matrices, synthetic CBF | Clean IBL cohorts, empirical psychometrics/CBFs, BIC tables, C1/C2 results |

---

## 3. The Day-1 Shared Interface Contract

To allow both tracks to build and test independently, both members commit to this exact interface on Day 1.

### 3.1 Standard Trial DataFrame Schema (`trials_df`)
Both synthetic generators and empirical loaders will output a `pandas.DataFrame` with identical columns:

```python
# Required Columns in trials_df:
# - subject_id:  str   (e.g., 'CSHL059' or 'sim_001')
# - trial_idx:   int   (0 to N-1)
# - contrast:    float (-1.0 to 1.0; signed contrast)
# - prior:       float (-1.0 for left-biased 20/80, +1.0 for right-biased 80/20)
# - prev_choice: int   (-1 for Left, +1 for Right, 0 for initial trial)
# - choice:      int   (-1 for Left boundary, +1 for Right boundary)
# - rt:          float (reaction time in seconds, e.g. 0.342)
```

### 3.2 Standard Model API
The fitting and simulation functions will follow an unchanging functional signature:

```python
# Simulation API (Track A builds, Track B can call)
def simulate_ddm(params: dict, model_type: str, contrasts: np.ndarray, priors: np.ndarray, prev_choices: np.ndarray) -> pd.DataFrame:
    """Returns trials_df with simulated choice and rt."""

# Fitting API (Track A builds, Track B uses for empirical fits)
def fit_ddm(trials_df: pd.DataFrame, model_type: str, bound_type: str = 'fixed', n_quantiles: int = 5) -> dict:
    """
    Fits DDM parameters via Quantile Maximum Probability (QMP).
    Returns dict: {'params': dict, 'nll': float, 'bic': float, 'success': bool}
    """
```

---

## 4. Week-by-Week Parallel Timeline

### Week 1: Independent Foundations & Mocking

#### Parth Dhodapkar (Track A)
- **Goal:** Build and verify the forward DDM simulation engine.
- **Tasks:**
  1. Implement vectorized Euler–Maruyama simulator in `src/models/simulator.py` ($\Delta t = 1\text{ ms}$, $\sigma = 1$).
  2. Implement bias mechanisms:
     - **M1 (Starting Point):** $x(0) = a \cdot z \cdot \text{prior}$
     - **M2 (Drift Bias):** $v(t) = k \cdot \text{sign}(c)|c|^\beta + v_h \cdot y_{t-1} + v_b \cdot \text{prior}$
     - **M3 (Dynamic Bias):** dynamic threshold/urgency $b(t) = \text{prior} \cdot b_{\max} \frac{t}{t + \tau_b}$
  3. Implement boundary dynamics: Fixed boundaries ($\pm a$) and Weibull collapsing boundaries $a(t) = a \exp(-(t/\lambda)^\gamma)$, with trial variabilities $s_v, s_z$.
  4. Generate synthetic verification plots (psychometric curves, RT distributions, Speed-Accuracy Trade-off).
- **Deliverable:** Working simulator module passing unit tests with reproducible psychometric/chronometric outputs.

#### Joshua Koilpillai (Track B)
- **Goal:** Ingest, parse, and clean IBL mouse behavioral data.
- **Tasks:**
  1. Set up `one-api` (`ibllib`) and connect to public IBL Alyx database.
  2. Query and select 10 trained mice from the biased protocol (alternating 20/80 and 80/20 blocks).
  3. Extract trial variables: signed contrast, block prior (20/80 vs 80/20), 1-back choice $y_{t-1}$, current choice, and RT ($t_{\text{wheel\_onset}} - t_{\text{stim\_onset}}$).
  4. Implement data exclusion filters: drop RT $< 0.1\text{ s}$ and $> 2.5\text{ s}$ (retain $5.0\text{ s}$ set for C2).
  5. Format and save local Parquet/CSV files conforming strictly to the §3.1 Trial DataFrame Schema.
- **Deliverable:** Cleaned dataset of 10 IBL mice saved in `data/processed/` with summary data statistics.

---

### Week 2: Independent Pipelines & Benchmarking

#### Parth Dhodapkar (Track A)
- **Goal:** Build QMP estimation engine and execute the **Wilson & Collins (2019)** recovery pipeline.
- **Tasks:**
  1. Implement Quantile Maximum Probability (QMP) loss across 5 RT quantiles $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ per contrast/choice condition in `src/optimization/qmp.py`.
  2. Implement Nelder–Mead optimization with common random numbers (CRN) for variance reduction.
  3. **Parameter Recovery (Rule 4):** Simulate 100 parameter vectors across trial counts $N \in \{200, 500, 1000, 2000\}$.
     - Fit models and calculate Pearson $r$, RMSE, linear slopes $\beta_1 \approx 1$.
     - Compute the parameter error correlation matrix $\text{Corr}(\Delta\theta_i, \Delta\theta_j)$ to prove $z$ and $v_b$ do not suffer from trade-offs/cross-talk.
  4. **Model Recovery (Rule 5):** Simulate 50 datasets per mechanism (M1, M2, M3); fit all three models to each; construct BIC confusion and inversion matrices.
- **Deliverable:** Automated recovery pipeline script and recovery figures showing $> 85\%$ model discriminability.

#### Joshua Koilpillai (Track B)
- **Goal:** Empirical behavioral characterization and diagnostic tool creation.
- **Tasks:**
  1. Compute empirical psychometric (sigmoid choice vs contrast) and chronometric (mean RT vs contrast) curves across all 10 mice.
  2. Implement diagnostic **Conditional Bias Function (CBF)** calculation (`src/empirical/cbf.py`):
     - Bin RTs into quintiles.
     - Calculate choice bias $\Delta P(\text{Right}) = P(\text{Right} \mid \text{Prior Right}) - P(\text{Right} \mid \text{Prior Left})$ as a function of RT quintile.
  3. Profile individual mouse heterogeneity across the 10 subjects.
  4. Build synthetic mock testing harness to verify that Track B's CBF and psychometric plotting tools correctly read any DataFrame adhering to the §3.1 Schema.
- **Deliverable:** Empirical baseline report (`notebooks/empirical_baseline.ipynb`) with complete psychometric, chronometric, and CBF figures for the 10 mice.

---

### Week 3: Integration & Empirical Hypothesis Testing (C1 & C2)

*At this stage, Parth's validated fitting engine is applied directly to Joshua's cleaned IBL dataset.*

#### Parth Dhodapkar (Focus: C1 Prior Mechanism Adjudication)
- **Goal:** Fit M1, M2, and M3 to the 10 IBL mice and adjudicate the computational route of the prior.
- **Tasks:**
  1. Execute fits of M1 ($z$), M2 ($v_b$), and M3 ($b(t)$) on each of the 10 IBL mice.
  2. Compute individual and aggregate $\Delta\text{BIC}$ tables relative to baseline (M2 vs M1 vs M3).
  3. Overlay model-predicted CBFs against Joshua's empirical CBFs:
     - M1 prediction: falling bias with RT.
     - M2 prediction: flat bias with RT.
     - M3 prediction: rising bias with RT.
  4. Quantify statistical significance (paired Wilcoxon / t-tests across mouse BICs).
- **Deliverable:** Core C1 results table and overlay plots adjudicating between starting point, drift bias, and dynamic bias.

#### Joshua Koilpillai (Focus: C2 Urgency & Boundary Dynamics)
- **Goal:** Fit fixed vs. collapsing bounds and test C1 mechanism invariance.
- **Tasks:**
  1. Fit Weibull collapsing boundary models ($a(t) = a \exp(-(t/\lambda)^\gamma)$) with trial variabilities $s_v, s_z$ to the 10 mice.
  2. Compare fixed vs. collapsing bounds under $\Delta\text{BIC}$ across standard ($2.5\text{ s}$) and extended ($5.0\text{ s}$) RT cutoffs.
  3. Analyze tail distributions to evaluate evidence for urgency in mouse wheel decisions.
  4. **Invariance Test:** Verify that the winning prior mechanism from C1 remains identical under collapsing boundaries.
- **Deliverable:** Core C2 results comparing fixed vs. collapsing bounds and proving prior route invariance.

---

### Week 4: Posterior Predictive Validation, Paper & Deliverables

#### Joint Collaboration
- **Posterior Predictive Checks (Rule 6):** Simulate full synthetic cohorts using the best-fit parameters from 10 mice and confirm qualitative reproduction of empirical choice/RT distributions.
- **Final Deliverables Split:**
  - **Parth:** Generate publication-grade figures (Vector graphics, parameter recovery correlation heatmaps, model confusion matrices, synthetic vs empirical CBFs). Clean and document the computational modeling repository.
  - **Joshua:** Draft primary report sections (Novelty & Background, IBL Task & Behavioral Preprocessing, Empirical Results, C1/C2 Discussion, Biological Implications).
  - **Joint Review:** Reconcile draft against proposal commitments, review code comments, assemble presentation slides.

---

## 5. Repository Structure & Git Collaboration Workflow

```text
incm_project/
├── data/
│   ├── raw/                 # Ignored by git; downloaded via ONE API
│   └── processed/           # Processed IBL CSV/Parquet conforming to §3.1
├── src/
│   ├── __init__.py
│   ├── models/              # [Track A - Parth]
│   │   ├── simulator.py     # Vectorized Euler–Maruyama (M1, M2, M3)
│   │   └── boundaries.py    # Fixed and Weibull collapsing boundaries
│   ├── optimization/        # [Track A - Parth]
│   │   ├── qmp.py           # Quantile Maximum Probability loss
│   │   └── fit.py           # Nelder–Mead optimization engine
│   ├── recovery/            # [Track A - Parth]
│   │   ├── param_recovery.py# Wilson & Collins Rule 4 pipeline
│   │   └── model_recovery.py# Wilson & Collins Rule 5 confusion matrix
│   ├── data/                # [Track B - Joshua]
│   │   ├── ibl_loader.py    # ONE API download & session filtering
│   │   └── preprocess.py    # RT filtering, feature extraction
│   ├── empirical/           # [Track B - Joshua]
│   │   ├── cbf.py           # Conditional Bias Function calculation
│   │   └── psychometrics.py # Empirical sigmoid & chronometric curves
│   └── visualization/       # [Shared]
│       ├── plots_recovery.py
│       └── plots_behavior.py
├── notebooks/
│   ├── 01_track_a_recovery_demo.ipynb
│   └── 02_track_b_empirical_baseline.ipynb
├── tests/
│   ├── test_simulator.py
│   └── test_data_contract.py
├── plan.md
└── proposal.md
```

### Git Branching Rules
1. `main` is always clean, tested, and runnable.
2. Track A works on `feature/sim-engine` and `feature/recovery-pipeline`.
3. Track B works on `feature/ibl-pipeline` and `feature/empirical-cbf`.
4. Merges into `main` require running `tests/test_data_contract.py` to ensure schema compatibility.

---

## 6. Contingency & Risk Management Matrix

| Risk / Failure Point | Probability | Impact | Responsible | Contingency Plan |
|---|---|---|---|---|
| **ONE API download is slow or fails** | Moderate | High | Joshua | Fall back to pre-downloaded public flat IBL tables (AWS S3 mirror) or pre-cached sessions from 5 mice. |
| **Nelder–Mead hits local minima on M3** | Moderate | Medium | Parth | Use 3 multi-start initializations with Latin Hypercube Sampling; fix $\tau_b$ on a coarse grid if needed. |
| **Collapsing bounds (C2) optimization slow** | Moderate | Low | Joshua / Joint | Per proposal contingency plan: report C2 as an exploratory robustness section without delaying C1. |
| **Parameter cross-talk between $z$ and $v_b$** | Low | Medium | Parth | Quantify in parameter recovery error correlation matrix $\text{Corr}(\Delta z, \Delta v_b)$; CBF curves provide independent qualitative dissociation. |

---

## 7. Immediate Next Steps (Day 1 Action Items)

1. **Parth:** Create repository directory structure (`src/`, `tests/`, `data/`) and implement `src/models/simulator.py` (M1, M2, M3 Euler–Maruyama).
2. **Joshua:** Install `one-api` (`pip install ONE-api ibllib`), test Alyx connection, and draft `src/data/ibl_loader.py` to pull the first biased session.
3. **Joint:** Run a smoke test confirming that Joshua's loader and Parth's simulator produce DataFrames that pass `tests/test_data_contract.py`.
