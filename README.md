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
│   ├── empirical/           # [Track B - Joshua] Empirical psychometrics & CBF
│   └── visualization/       # Publication-grade plotting routines
├── tests/
│   ├── test_data_contract.py# Schema validation & physical bounds
│   └── test_simulator.py    # DDM physics & loss surface tests
├── run_track_a.py           # CLI benchmark harness for Track A
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
