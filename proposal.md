# Project Proposal: How Does an Uncued, Inferred Prior Enter the Decision Process? A Drift Diffusion Model Study of Mouse Perceptual Choice

**Authors:** Parth Dhodapkar (2024111009), Joshua Koilpillai (2024101128)  
**Course:** Introduction to Neural and Cognitive Modeling (INCM) | **Topic:** Evidence Accumulation (Question 27)

---

## 1. Project Scope & Research Questions

When deciding between two sensory alternatives, the brain accumulates noisy sensory evidence over time until crossing an internal decision boundary. The Drift Diffusion Model (DDM; Ratcliff, 1978; Bogacz et al., 2006) captures this process, accounting for accuracy, full response time (RT) distributions, and the speed-accuracy trade-off (SAT).

Decisions are also shaped by prior expectations, which can enter the DDM via three distinct computational routes:
1. **Starting Point Bias (z):** Shifts initial accumulator position toward the expected boundary before evidence begins (pre-evidence motor readiness).
2. **Drift Bias (v_b):** Adds an offset to the evidence accumulation rate (asymmetric sensory evidence weighting).
3. **Dynamic Bias (b(t)):** Grows with elapsed decision time, creating an urgency-dependent reliance on the prior when sensory evidence is ambiguous (Hanks et al., 2011).

Recent literature suggests that the route depends on how the prior is generated:
- **Explicit, cued priors** (e.g., instructions that the target is 80% likely right) shift the **starting point** (Mulder et al., 2012; Hellmann & Forster, 2026).
- **Implicit sequential biases** from the immediate previous choice shift the **drift rate** across humans and rodents (Urai et al., 2019; Fernandez et al., 2025).

### The Gap and Theoretical Dilemma
In ecological environments, statistical regularities are **neither explicitly cued nor reducible to 1-back motor persistence**. Instead, animals must infer latent statistical states from sequential experience. In the International Brain Laboratory (IBL, 2021) 2AFC visual task, mice experience blocks of 80/20 and 20/80 left/right stimuli without any explicit cues. While Findling et al. (2023) confirmed mice infer this prior and represent it brain-wide, they analyzed static choice probabilities without a mechanistic evidence-accumulation model. Strikingly, Fernandez, Fengler & Urai (2025) fit DDMs across species to study choice history in IBL mice, but **deliberately excluded all sessions with biased probability blocks**. Thus, how an inferred, uncued prior enters the decision process remains completely unaddressed.

### Computational Modeling Protocol (Wilson & Collins, 2019)
Following feedback, rather than introducing a superficial second dataset, we ground our project in a **comprehensive parameter recovery and model recovery investigation** adhering to the rigorous guidelines of **Wilson & Collins (2019)** (*"Ten simple rules for the computational modeling of behavioral data"*, *eLife*). Before fitting empirical data, we establish parameter identifiability, quantify estimation error, rule out parameter cross-talk (e.g., *z* vs. *v_b* trade-offs), and determine the statistical power required across mouse trial counts.

### Project Structure & Contributions
- **Part A: Baseline Reproduction & Wilson & Collins Validation:**
  1. Build a vectorized Euler–Maruyama DDM simulator in Python (`numpy`, `scipy`, `matplotlib`); reproduce psychometric and chronometric curves and the SAT.
  2. Conduct comprehensive parameter recovery across plausible parameter space (*k, a, t₀, z, v_b, v_h, b_max, τ_b*) at varying trial counts (*N* = 200, 500, 1000, 2000). Quantify recovery correlations (*r*), RMSE, linear slopes, and parameter error correlation matrices.
  3. Run model recovery across competing models to construct confusion and inversion matrices under BIC, confirming absence of model mimicry.
- **Part B: Core Scientific Contributions:**
  - **C1 (Primary Contribution): Route of the Inferred Prior.** Fit three competing mechanisms to IBL mice in biased blocks, controlling for 1-back choice history:
    - **M1 (Starting Point):** *x*(0) = *a* · *z* · prior → bias strongest for **fast** RTs, then **declines**.
    - **M2 (Drift Bias):** *v*(*t*) = *v*_sensory + *v_b* · prior + *v_h* · *y*_(t-1) → bias is **flat** across RTs.
    - **M3 (Dynamic Bias):** *x*_eff(*t*) = *x*(*t*) + prior · *b*_max · *t* / (*t* + *τ_b*) → bias **increases** for slow RTs.  
    Adjudicate between M1–M3 using quantitative model comparison (ΔBIC) alongside the diagnostic **Conditional Bias Function (CBF)** (White & Poldrack, 2014; Wilson & Collins Rule 6).
  - **C2 (Secondary Contribution): Urgency & Collapsing Bounds on the Wheel Task.** Compare fixed vs. Weibull-collapsing boundaries, controlling for across-trial drift (*s_v*) and starting-point (*s_z*) variabilities which can mimic urgency (Hawkins et al., 2015). Verify that the C1 prior mechanism is invariant to boundary dynamics.

---

## 2. Novelty Statement

| Prior Literature | What It Did | What It Left Open | Our Distinct Contribution |
|---|---|---|---|
| **Fernandez et al. (2025)** | DDM on IBL mice; choice history → drift | Excluded biased-block sessions | First DDM fit to IBL **biased-block protocol** |
| **Findling et al. (2023)** | Inferred prior decoded across mouse brain | Static logistic regression; no DDM | **Dynamic accumulation model** of inferred priors |
| **Mulder et al. (2012)** | Cued prior → starting point *z* | Used explicit cues in humans | Tests **uncued, inferred** prior in rodents |
| **Hanks et al. (2011)** | Dynamic time-growing prior bias | Macaque neurophysiology; primates only | First test of dynamic prior bias in **mice** |
| **Chakravarty et al. (2024)** | Collapsing bounds in rodent nose-pokes | Continuous wheel tasks untested | Tests fixed vs. collapsing bounds on **wheel task** |
| **Wilson & Collins (2019)** | 10 rules for computational modeling | Often omitted in empirical DDM studies | **Comprehensive parameter/model recovery pipeline** |

---

## 3. Planned Datasets & Methodologies

### 3.1 Datasets
1. **IBL Mouse Dataset (2021):** Open-access data via `one-api`. Head-fixed mice turn a wheel to report grating side at contrasts 0%, 6.25%, 12.5%, 25%, 50%, 100%. We select 10 trained mice from the biased protocol (alternating 20/80 and 80/20 blocks). For each trial, we extract signed contrast, block prior, previous choice, current choice, and RT = t_wheel_onset − t_stimulus_onset. Trials with RT < 0.1 s or > 2.5 s are excluded (5.0 s cutoff for C2).
2. **Synthetic Benchmarks:** Grids of synthetic sessions across parameter ranges (*k, a, t₀, z, v_b, v_h, b_max, τ_b*) and trial counts (*N* ∈ {200, 500, 1000, 2000}).

### 3.2 Computational Model Family & QMP Fitting
Accumulation follows *dx*(*t*) = *v*(*t*)*dt* + *σ dW*(*t*) (*σ* = 1, Δ*t* = 1 ms) with boundaries at ±*a*. Drift is *v*(*t*) = *k* · sign(*c*)|*c*|^β^ + *v_h* · *y*_(t-1) + [*v_b* · prior]_M2. M1 shifts *x*(0) = *a* · *z* · prior; M3 adds dynamic bias prior · *b*_max · *t* / (*t* + *τ_b*); C2 introduces Weibull collapse *a*(*t*) = *a* · exp(−(*t*/*λ*)^γ^) and trial variabilities *s_v*, *s_z*.

Parameters are fit using Quantile Maximum Probability (QMP; Heathcote et al., 2002) across five RT quantiles {0.1, 0.3, 0.5, 0.7, 0.9} per contrast/choice condition, optimized via Nelder–Mead with fixed common random numbers.

### 3.3 Wilson & Collins (2019) Parameter & Model Recovery Pipeline
- **Parameter Recovery (Rule 4):** Fit synthetic datasets across 100 parameter vectors. Plot ground-truth vs. recovered parameters; quantify Pearson *r*, RMSE, and linear slopes (*β*₁ ≈ 1). Compute the recovery error correlation matrix Corr(Δ*θ*_i, Δ*θ*_j) to guarantee that *z* and *v_b* are statistically dissociable without trade-offs.
- **Model Recovery (Rule 5):** Simulate 50 datasets per mechanism (M1, M2, M3), fit all competing models to each dataset, and compute confusion/inversion matrices via BIC to verify > 85% discriminability.
- **Posterior Predictive Checks (Rule 6):** Simulate behavior from fitted parameters to confirm qualitative reproduction of empirical psychometric, chronometric, and CBF curves.

---

## 4. Evaluation Strategies & Timeline

### Evaluation Strategies
1. **Recovery Metrics:** Scatterplots of true vs. recovered parameters, recovery correlation *r* > 0.8, and confusion matrix diagonal dominance.
2. **Baseline Fits:** Empirical vs. simulated psychometric (sigmoid) and chronometric curves across contrasts.
3. **Prior Mechanism (C1):** Individual and summed ΔBIC across 10 mice; empirical CBF overlaid against M1 (falling), M2 (flat), and M3 (rising) predictions.
4. **Urgency & Invariance (C2):** ΔBIC for fixed vs. collapsing bounds (with *s_v*, *s_z* controls), tail distribution analysis at 5.0 s cutoff, and confirmation of C1 outcome consistency.

### Timeline & Work Distribution (2 People, 4 Weeks)

| Week | Parth Dhodapkar | Joshua Koilpillai | Key Deliverables |
|---|---|---|---|
| **1** | Implement vectorized simulator (M1–M3, collapsing bounds) | Download IBL dataset via ONE API; filter trials, extract RT/labels | Verified simulator; clean IBL dataset |
| **2** | Wilson & Collins Parameter Recovery pipeline (*N* = 200–2000) | QMP fitting pipeline; Model Recovery confusion matrix; baseline fits | Parameter & model recovery benchmarks |
| **3** | **C1:** Fit M1–M3 to 10 IBL mice; BIC analysis; compute CBF curves | **C2:** Fixed vs. collapsing fits (*s_v*, *s_z* controls); test C1 invariance | Mechanism adjudication (C1); boundary results (C2) |
| **4** | Posterior predictive validation; generate publication-grade figures | Draft report sections (Novelty, Recovery, Findings); code cleanup | Final report, figures, and code repository |

*Contingency Plan:* C1 and the Wilson & Collins parameter recovery investigation are the core protected deliverables. If collapsing bounds (C2) encounter computational bottlenecks, C2 will be reported as an exploratory robustness subsection.

---

## 5. Key References

1. **Bogacz, R., et al.** (2006). The physics of optimal decision making. *Psychol Rev*, 113(4), 700–765.
2. **Chakravarty, S., et al.** (2024). A cross-species framework for investigating perceptual evidence accumulation. *bioRxiv*, 10.1101/2024.04.17.589945.
3. **Fernandez, K., Fengler, A., & Urai, A. E.** (2025). Choice history biases evidence accumulation. *bioRxiv*, 10.1101/2025.08.22.671879.
4. **Findling, C., et al.** (2023). Brain-wide representations of prior information in mouse decision-making. *bioRxiv*, 10.1101/2023.07.04.547684.
5. **Hanks, T. D., et al.** (2011). Elapsed decision time affects the weighting of prior probability. *J Neurosci*, 31(17), 6339–6352.
6. **Hawkins, G. E., et al.** (2015). Revisiting evidence for collapsing boundaries. *J Neurosci*, 35(6), 2476–2484.
7. **Heathcote, A., et al.** (2002). Quantile maximum likelihood estimation of RT distributions. *Psychon Bull Rev*, 9(2), 394–401.
8. **Hellmann, S., & Forster, C.** (2026). Two paths to bias. *Comput Brain Behav*, 10.1007/s42113-026-00331-6.
9. **IBL.** (2021). Standardized measurement of decision-making in mice. *eLife*, 10, e63711.
10. **Mulder, M. J., et al.** (2012). Bias in the brain: Diffusion analysis of prior probability. *J Neurosci*, 32(7), 2335–2343.
11. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model. *Neural Comput*, 20(4), 873–922.
12. **Urai, A. E., et al.** (2019). Choice history biases subsequent evidence accumulation. *eLife*, 8, e46331.
13. **White, C. N., & Poldrack, R. A.** (2014). Decomposing bias in simple decisions. *J Exp Psychol Learn Mem Cogn*, 40(2), 385–398.
14. **Wilson, R. C., & Collins, A. G.** (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*, 8, e49547.
