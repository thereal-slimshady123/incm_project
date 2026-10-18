# Project Proposal: How Does an Uncued, Inferred Prior Enter the Decision Process? A Drift Diffusion Model Study of Mouse Perceptual Choice

**Authors:** Parth Dhodapkar (2024111009), Joshua Koilpillai (2024101128)  
**Course:** Introduction to Neural and Cognitive Modeling (INCM) | **Topic:** Evidence Accumulation (Question 27)

---

## 1. Project Scope & Research Questions

When deciding between two sensory alternatives—such as whether a visual grating appeared on the left or right of a display—the brain does not base its choice on a single snapshot. Instead, it accumulates noisy sensory evidence over time until reaching an internal decision threshold. The Drift Diffusion Model (DDM; Ratcliff, 1978; Bogacz et al., 2006) provides the foundational mathematical framework for this process, capturing accuracy, full response time (RT) distributions, and the speed-accuracy trade-off (SAT).

Natural decisions, however, are rarely made in a statistical vacuum. What an agent expects prior to stimulus presentation heavily biases the subsequent choice. In the DDM framework, prior expectations can enter the accumulation architecture through three distinct computational mechanisms:
1. **Starting Point Bias ($z$):** Shifting the initial position of the accumulator toward the anticipated boundary prior to stimulus onset, representing static pre-evidence motor or decision preparation.
2. **Drift Bias ($v_b$):** Adding a constant offset to the evidence accumulation rate, representing a top-down subjective weighting or asymmetric filtering of incoming sensory evidence.
3. **Dynamic Bias ($b(t)$):** An offset that grows dynamically with elapsed decision time, representing an urgency-dependent reliance on prior expectations when ambiguous sensory evidence fails to resolve the decision.

Recent literature demonstrates that the route an expectation takes depends critically on the **nature and origin of the prior**:
- **Explicit, cued priors** (e.g., visual or verbal cues instructing human subjects that the target is 80% likely to appear on the right) predominantly shift the **starting point** (Mulder et al., 2012; Hellmann & Forster, 2026).
- **Implicit sequential biases** arising from the immediate previous choice (1-back choice history) modulate the **drift rate** across humans and rodents (Urai et al., 2019; Fernandez et al., 2025).

### The Critical Gap: Uncued, Dynamically Inferred Priors
In naturalistic ecological foraging, priors are **neither explicitly signaled nor reducible to simple 1-back motor persistence**. Instead, animals must continuously infer the latent statistical structure of their environment from sequential feedback over extended horizons. 

In the standardized International Brain Laboratory (IBL, 2021) mouse task, visual stimuli appear on one side with 80% probability for an unsignaled block of trials, which then abruptly switches to 20% or 50%. Mice are **never explicitly cued** about the current block; they must dynamically infer the underlying state. While recent neurobiological investigations confirm that mice maintain an internal estimate of this block prior and represent it across wide cortical circuits (Findling et al., 2023), **how this inferred latent belief interfaces with the millisecond-scale evidence accumulation process remains entirely unknown**. Crucially, the landmark cross-species DDM study by Fernandez, Fengler, and Urai (2025) fit drift diffusion models to IBL mice but **deliberately excluded all sessions containing biased probability blocks**, leaving the computational routing of inferred priors completely unaddressed.

### Computational Modeling Protocol (Wilson & Collins, 2019)
To resolve this question conclusively, modeling behavioral data requires more than curve fitting. As formalized by **Wilson & Collins (2019)** in *"Ten simple rules for the computational modeling of behavioral data"*, complex cognitive models with multiple bias mechanisms, parameter variabilities, and boundary dynamics are vulnerable to parameter collinearity, identifiability failure, and model mimicry. Following feedback, rather than introducing a superficial second dataset with disparate task dynamics, we ground this project in a **comprehensive parameter recovery and model recovery investigation** following Wilson & Collins' protocol. We rigorously demonstrate that our parameter estimation pipeline uniquely recovers ground-truth parameters, rules out parameter crosstalk (e.g., separating starting point bias from drift bias), and establishes statistical power across mouse trial counts before fitting empirical animal data.

### Project Structure

#### Part A: Foundation, Computational Rigor & Baseline Reproduction
1. **Simulator & Baseline Reproduction:** Implement an Euler–Maruyama DDM simulator in Python (`numpy`, `scipy`, `matplotlib`). Reproduce canonical psychometric (accuracy vs. contrast) and chronometric (RT vs. contrast) curves, and simulate the speed-accuracy trade-off by manipulating threshold height $a$.
2. **Comprehensive Parameter Recovery Investigation (Wilson & Collins, 2019, Rules 3 & 4):** Synthesize benchmark datasets across wide parameter ranges. Quantify parameter identifiability, estimation bias, recovery correlations ($r$), and recovery error (RMSE). Compute the correlation matrix of recovery errors across parameters to test for parameter trade-offs (e.g., $z$ vs. $v_b$). Evaluate parameter recovery fidelity across varying trial counts ($N = 200, 500, 1000, 2000$) to determine the trial count required for reliable inference.
3. **Model Recovery & Confusion Analysis (Wilson & Collins, 2019, Rule 5):** Simulate synthetic sessions from each candidate model, refit all competing models, and construct the model confusion and inversion matrices under Bayesian Information Criterion (BIC) to verify that the mechanisms do not mimic one another.

#### Part B: Core Scientific Contributions
- **Contribution 1 (C1, Primary Novelty): Identifying the Route of the Inferred Prior.** Fit three competing DDM mechanisms to empirical IBL mouse choices and RT distributions during biased blocks, including an explicit 1-back choice history drift term to dissociate block-level inferred priors from local choice repetition:
  
  | Mechanism | Mathematical Implementation | Predicted Conditional Bias vs. RT | Theoretical Interpretation |
  |---|---|---|---|
  | **M1: Starting Point Bias** | $x(0) = z \cdot \text{prior} \cdot a$ | Strongest on **fast** trials, rapidly **declines** | Mice treat inferred prior as pre-evidence motor readiness, identical to explicit cued priors |
  | **M2: Drift Rate Bias** | $v(t) = v_{\text{sensory}} + v_b \cdot \text{prior}$ | **Flat** across RT quantiles | Inferred prior acts as top-down sensory weighting, identical to choice history |
  | **M3: Dynamic Elapsed Bias** (Hanks et al., 2011) | $x(t) \to x(t) + \text{prior} \cdot b_{\max} \frac{t}{t + \tau_b}$ | **Increases** for slow, difficult trials | Mice fall back on the inferred prior when sensory evidence fails to terminate deliberation |

  We adjudicate between M1–M3 using quantitative model selection (BIC) alongside qualitative diagnostic signatures via the **Conditional Bias Function (CBF)** (White & Poldrack, 2014; Wilson & Collins, 2019, Rule 6).

- **Contribution 2 (C2, Secondary): Collapsing Bounds and Urgency on the IBL Wheel Task.** Test whether decision bounds collapse over time in mice turning a physical wheel. While collapsing bounds were recently reported in rodents performing head-fixed nose-pokes (Chakravarty et al., 2024), they have not been established on the IBL wheel task. We compare fixed vs. Weibull-collapsing boundaries, controlling for across-trial drift rate variability ($s_v$) and starting point variability ($s_z$) which are known to mimic urgency (Hawkins et al., 2015). We verify that the prior mechanism identified in C1 is invariant to boundary dynamics.

### Core Research Questions
1. **Identifiability & Recovery (Wilson & Collins Rules 4–5):** Are the parameters of competing prior-bias DDM mechanisms (starting point, drift bias, dynamic bias) uniquely recoverable without parameter crosstalk, and does standard IBL mouse session length provide adequate statistical power?
2. **Mechanistic Routing (C1, Core Question):** When animals dynamically infer an uncued environmental prior from experience, does the prior shift the starting point ($z$), bias evidence accumulation ($v_b$), or enter as a time-growing dynamic bias ($b(t)$)?
3. **Urgency & Boundary Dynamics (C2):** Do mouse wheel-turning decisions exhibit collapsing decision thresholds once across-trial parameter variabilities ($s_v, s_z$) are accounted for, and does the winning prior-bias mechanism replicate under both boundary geometries?

---

## 2. Novelty Statement & Theoretical Rationale

### Why This Research Question is Substantive and Novel
The cognitive neuroscience of perceptual decision-making has focused extensively on two extreme ends of expectation:
1. **Explicit, Cued Priors:** When human participants are explicitly instructed via visual or verbal cues before each trial (e.g., "70% probability of Right"), the decision process predominantly shifts the starting point $z$ toward the cued alternative (Mulder et al., 2012; Hellmann & Forster, 2026). This reflects an active, conscious, pre-stimulus motor preparation.
2. **Immediate Choice History:** When organisms exhibit sequential choice biases (repeating or alternating recent choices), this unconscious motor persistence or reinforcement habit shifts the drift rate $v_b$ (Urai et al., 2019; Fernandez et al., 2025).

**The Missing Domain:** Ecological decision-making almost never provides explicit instructional cues. Animals must infer environmental statistics from ongoing experience across tens or hundreds of trials. Is an internally computed, inferred statistical prior treated as an **instructed motor bias** (starting point $z$), an **accumulated sensory filter** (drift bias $v_b$), or an **urgency-dependent fallback** (dynamic bias $b(t)$)? This represents a fundamental question regarding how latent belief states interface with evidence accumulation.

### Positioning Relative to Closest Literature

| Study | Paradigm & Model | Key Finding | What It Left Open | Our Added Contribution |
|---|---|---|---|---|
| **Fernandez, Fengler & Urai (2025)** | DDM on IBL mice, humans & monkeys | Choice history biases drift rate across species | **Deliberately excluded all biased-block sessions** from IBL mouse data | First study to fit evidence accumulation models to the **biased-block protocol** of IBL mice |
| **Findling et al. (2023)** | Logistic regression & GLM-HMM on IBL mice | Mice track block priors; represented brain-wide | Analyzed static choice probability; **no mechanistic DDM** of RTs or accumulation | Provides the **mechanistic accumulation account** linking inferred priors to RT distributions |
| **Mulder et al. (2012); Hellmann & Forster (2026)** | DDM on humans with explicit cues | Cued probability shifts starting point $z$ | Priors were **explicitly cued**, not inferred from experience | Determines whether an **inferred, uncued** prior behaves like an explicit cue |
| **Hanks et al. (2011)** | Dynamic bias DDM in macaque neurophysiology | Prior probability acts as a time-growing bias signal | Observed in overtrained primates; **never tested in rodents** | Tests whether dynamic, time-growing prior bias operates in mouse perceptual decisions |
| **Chakravarty et al. (2024)** | DDM with collapsing bounds in rats & mice | Collapsing boundaries fit rodent nose-poke tasks | Tested only discrete nose-pokes; **wheel tasks untested** | Tests fixed vs. collapsing bounds on continuous wheel choices with $s_v, s_z$ controls |
| **Wilson & Collins (2019)** | Methodological framework for cognitive modeling | Outlines 10 simple rules for parameter & model recovery | Parameter recovery often neglected in empirical DDM studies | **Comprehensive parameter recovery & confusion analysis** before empirical fitting |

### Why Comprehensive Parameter Recovery Replaces a Superficial Second Dataset
A common recommendation in computational modeling is either to analyze an additional dataset or to conduct a comprehensive parameter recovery investigation (Wilson & Collins, 2019). We deliberately execute the comprehensive parameter recovery investigation for clear scientific and methodological reasons:
- **Identifiability Risk:** In multi-parameter diffusion models that incorporate baseline drift, history bias, block bias, boundary heights, and non-decision time, models can easily overfit or suffer from parameter trade-offs (e.g., trade-offs between starting point $z$ and drift bias $v_b$, or between boundary height $a$ and drift rate $v$).
- **Statistical Power Validation:** Rodent datasets have distinct trial counts per session (~300–800 trials). Applying a model to real data without proving that its parameters can be recovered at those specific trial counts undermines the validity of any conclusions.
- **Scientific Rigor:** Performing systematic parameter recovery across synthetic grids, plotting ground-truth vs. recovered parameters, quantifying recovery errors (RMSE, $r$, regression slope), and constructing model confusion matrices provides the rigorous methodological foundation advocated by Wilson & Collins (2019), elevating the project from descriptive fitting to robust computational modeling.

<div style="page-break-after: always;"></div>

---

## 3. Literature Review

### 3.1 Foundations of Evidence Accumulation and the Drift Diffusion Model
The Drift Diffusion Model (Ratcliff, 1978; Ratcliff & McKoon, 2008) formalizes two-alternative forced-choice (2AFC) decisions as continuous accumulation of noisy sensory evidence toward one of two decision thresholds. Bogacz et al. (2006) proved that the flat-boundary DDM implements the continuous-time Sequential Probability Ratio Test (SPRT), optimizing mean decision time for a fixed error rate. Across-trial variabilities in drift rate ($s_v$) and starting point ($s_z$) capture slow and fast error distributions, respectively (Ratcliff & Tuerlinckx, 2002).

### 3.2 Computational Mechanisms of Prior and History Biases
Decisions are biased by both recent history and environmental statistics:
- **Starting Point vs. Drift Rate:** Summerfield & de Lange (2014) and Mulder et al. (2012) showed that explicit prior cues primarily adjust the starting point $z$, allowing agents to pre-emptively shift closer to the expected boundary. In contrast, Urai et al. (2019) demonstrated across multiple human datasets that choice history biases the drift rate $v$, reflecting a post-decisional modulation of sensory evidence processing. Hellmann & Forster (2026) confirmed that stimulus probability and previous choice operate through distinct channels in humans.
- **Dynamic Bias Signals:** In neurophysiological recordings of macaque lateral intraparietal area (LIP), Hanks et al. (2011) showed that prior probability is not a static offset; instead, the neural representation of prior bias grows over the course of a single trial. White & Poldrack (2014) introduced the **Conditional Bias Function (CBF)** as a non-parametric behavioral diagnostic: starting point shifts produce biases that are maximal on fast trials and decay toward zero, whereas drift biases produce flat CBFs across RTs, and dynamic biases produce rising CBFs.

### 3.3 Urgency and Boundary Dynamics
Whether decision boundaries remain constant or collapse over time remains actively debated. Normative theories suggest that when trial duration is costly or stimulus difficulty is unknown, collapsing thresholds optimize reward rate (Drugowitsch et al., 2012; Cisek et al., 2009). While Hawkins et al. (2015) argued that fixed boundaries with across-trial variability adequately account for human and primate data, recent findings by Chakravarty et al. (2024) observed that collapsing bounds provide superior fits to rodent evidence accumulation in nose-poke tasks. Testing whether this holds on the IBL wheel-turning paradigm is critical for characterizing rodent decision dynamics.

### 3.4 Wilson & Collins (2019) Modeling Best Practices
Wilson & Collins (2019) established guidelines for behavioral modeling:
- **Rule 2 & 3:** Explicit mathematical formulation and rigorous optimization.
- **Rule 4 (Parameter Recovery):** Simulating synthetic data from known parameters across physiological parameter space and refitting to demonstrate parameter identifiability ($r > 0.8$, minimal bias).
- **Rule 5 (Model Recovery):** Fitting all candidate models to synthetic data generated from each model to verify that models can be distinguished and do not suffer from mimicry.
- **Rule 6 (Model Validation):** Demonstrating that fitted models generate synthetic behavioral curves (psychometric, chronometric, CBF) that qualitatively match empirical data.

---

## 4. Planned Datasets & Synthetic Benchmarks

### 1. Empirical Mouse Dataset: International Brain Laboratory (IBL, 2021)
- **Source:** Publicly available IBL data accessed via the open `one-api` Python package. We collect no animal data ourselves; all analysis is computational.
- **Task Description:** Head-fixed mice turn a LEGO wheel to indicate whether a sinusoidal grating visual stimulus appeared on the left or right. Grating contrasts span $0\%, 6.25\%, 12.5\%, 25\%, 50\%, 100\%$. Correct choices yield a sweetened water reward; incorrect choices yield a white noise burst and a 2-second timeout.
- **Biased Block Protocol:** Sessions follow a block structure: an initial unbiased block (50/50 probability, 90 trials) followed by alternating blocks of 20/80 and 80/20 probability of rightward vs. leftward targets (block lengths 20–100 trials, sampled from a truncated exponential distribution).
- **Cohort & Filtering:** We select 10 fully trained mice exhibiting stable performance on the biased protocol (~1,000–3,000 trials per mouse across multiple sessions). For every trial, we extract: stimulus contrast $c$, stimulus side, block probability $P(\text{Right})$, previous trial choice $y_{t-1}$, current choice $y_t$, and response time ($\text{RT} = t_{\text{wheel onset}} - t_{\text{stimulus onset}}$).
- **Exclusion Criteria:** Anticipatory trials ($\text{RT} < 0.1\,\text{s}$) and disengaged trials ($\text{RT} > 2.5\,\text{s}$) are filtered out. For boundary collapse analysis (C2), we re-examine with a 5.0 s cutoff to preserve the distribution tail.

### 2. Synthetic Benchmark Datasets for Parameter & Model Recovery (Wilson & Collins, 2019)
- **Grid Generation:** Synthetic datasets generated across realistic parameter ranges:
  - Drift sensitivity $k \in [1.0, 5.0]$, non-linear contrast exponent $\beta \in [0.5, 1.2]$
  - Baseline boundary height $a \in [0.5, 2.0]$
  - Non-decision time $t_0 \in [0.10, 0.35]\,\text{s}$
  - Starting point prior weight $z \in [-0.4, +0.4]$
  - Drift prior weight $v_b \in [-1.5, +1.5]$
  - History drift weight $v_h \in [-1.0, +1.0]$
  - Dynamic bias parameters $b_{\max} \in [0.5, 2.5]$, $\tau_b \in [0.1, 0.6]\,\text{s}$
  - Across-trial variabilities $s_v \in [0.1, 0.8]$, $s_z \in [0.05, 0.3]$
  - Weibull collapse parameters $\lambda \in [0.5, 2.5]\,\text{s}$, shape $\gamma \in [1.0, 3.5]$
- **Sample Size Scaling:** Simulations generated at trial counts $N \in \{200, 500, 1000, 2000\}$ to establish empirical power curves for parameter recovery.

---

## 5. Methodologies

### 5.1 Model Family & Formal Equations
All model variants share an evidence accumulation counter $x(t)$ governed by a 1D diffusion process:
$$dx(t) = v(t)\,dt + \sigma\,dW(t), \quad x(0) = x_0, \quad \sigma = 1$$
In discrete simulation (Euler–Maruyama, step size $\Delta t = 1\,\text{ms}$):
$$x(t + \Delta t) = x(t) + v(t)\,\Delta t + \sigma \sqrt{\Delta t}\,\xi(t), \quad \xi(t) \sim \mathcal{N}(0, 1)$$
Decision time $T_d$ is the first passage time where $\lvert x(t)\rvert \ge a(t)$. Total observed response time is $\text{RT} = T_d + t_0$.

Evidence drift $v(t)$ incorporates sensory input and choice history across all models:
$$v_{\text{sensory}} = k \cdot \text{sign}(c) \cdot \lvert c\rvert^{\beta}$$
$$v_{\text{history}} = v_h \cdot y_{t-1}, \quad y_{t-1} \in \{-1, +1\}$$
where $c$ is signed contrast ($c \in \{-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, \dots, +1\}$).

The block prior is coded as $\text{prior} \in \{-1, 0, +1\}$ for 20%, 50%, and 80% rightward blocks. The competing prior mechanisms are implemented as follows:
- **Model 1 (M1, Starting Point Bias):**
  $$x_0 = a \cdot z \cdot \text{prior}, \quad v(t) = v_{\text{sensory}} + v_{\text{history}}$$
- **Model 2 (M2, Drift Bias):**
  $$x_0 = 0, \quad v(t) = v_{\text{sensory}} + v_{\text{history}} + v_b \cdot \text{prior}$$
- **Model 3 (M3, Dynamic Elapsed Bias):**
  $$x_0 = 0, \quad v(t) = v_{\text{sensory}} + v_{\text{history}}, \quad x_{\text{eff}}(t) = x(t) + \text{prior} \cdot b_{\max} \frac{t}{t + \tau_b}$$
  with termination when $\lvert x_{\text{eff}}(t)\rvert \ge a$.

For Contribution 2 (C2), we add across-trial Gaussian drift variability $v \sim \mathcal{N}(v_{\text{mean}}, s_v^2)$, uniform starting-point variability $x_0 \sim \mathcal{U}(-s_z a, s_z a)$, and evaluate the Weibull collapsing boundary:
$$a(t) = a \cdot \exp\left(-\left(\frac{t}{\lambda}\right)^\gamma\right)$$

### 5.2 Simulation & Quantile Maximum Probability (QMP) Fitting
Because dynamic bias (M3) and collapsing boundaries preclude simple analytical closed-form likelihood solutions, all candidate models are fit using a unified, simulation-based optimization approach:
1. **Simulation:** Vectorized Monte Carlo integration ($\Delta t = 1\,\text{ms}$, max duration $5.0\,\text{s}$, 10,000 trajectories per condition).
2. **Quantile Estimation:** For each contrast level and choice option, empirical RT distributions are summarized by five quantiles: $q \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$, dividing RT space into six bins.
3. **Objective Function:** We minimize the Quantile Maximum Probability (QMP) loss (Heathcote et al., 2002; Ratcliff & Tuerlinckx, 2002), equivalent to multinomial log-likelihood:
   $$\ln L_{\text{QMP}} = \sum_{c} \sum_{y \in \{L, R\}} \sum_{i=1}^{6} N_{c, y, i} \ln P_{\text{model}}(c, y, i \mid \theta)$$
   where $N_{c, y, i}$ is observed trial counts in bin $i$ and $P_{\text{model}}$ is simulated probability mass.
4. **Optimization:** Nelder–Mead simplex algorithm with common random numbers (CRNs) fixed across iterations to ensure a deterministic, smooth objective function and avoid stochastic chatter.

### 5.3 Comprehensive Parameter & Model Recovery Protocol (Wilson & Collins, 2019)

```
+--------------------------------------------------------------------------+
|             Wilson & Collins (2019) Computational Modeling Pipeline       |
+--------------------------------------------------------------------------+
|  1. Model Formulation (Rule 2)                                           |
|     Explicit equations for M1 (z), M2 (v_b), M3 (dynamic bias)           |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|  2. Parameter Recovery (Rule 4)    |  3. Model Recovery (Rule 5)         |
|     * Sample parameter grid        |     * Simulate from M1, M2, M3      |
|     * Fit synthetic data via QMP   |     * Fit all models to each dataset|
|     * Plot True vs. Recovered      |     * Construct Confusion Matrix    |
|     * Pearson r, RMSE, Bias        |     * Inversion Matrix P(gen|fit)   |
|     * Parameter Correlation Matrix |     * Confirm discriminability      |
|     * Power vs. Trial Count (N)    |                                     |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|  4. Empirical Fitting & Model Selection (Rules 3 & 7)                    |
|     * Fit M1, M2, M3 to 10 IBL mice                                      |
|     * BIC & AIC model comparison (individual & group-level)              |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|  5. Model Validation via Posterior Predictive Checks (Rule 6)            |
|     * Simulate choices and RTs from fitted parameters                    |
|     * Validate Psychometric, Chronometric, and Conditional Bias Curves   |
+--------------------------------------------------------------------------+
```

1. **Parameter Recovery Procedure (Rule 4):**
   - Sample 100 parameter vectors uniformly from plausible ranges.
   - For each parameter vector, simulate synthetic datasets at $N = 200, 500, 1000, 2000$ trials.
   - Run our QMP Nelder-Mead fitting pipeline to estimate recovered parameters $\hat{\theta}$.
   - Compute the Pearson correlation coefficient $r(\theta, \hat{\theta})$ and root mean square error $\text{RMSE}(\theta, \hat{\theta})$ for each parameter ($k, a, t_0, z, v_b, v_h, b_{\max}, \tau_b$).
   - Compute linear regression slopes $\hat{\theta} = \beta_1 \theta + \beta_0$ to test for regression-to-the-mean or systematic shrinkage ($\beta_1 \approx 1$, $\beta_0 \approx 0$).
   - Compute the **parameter cross-talk matrix**: correlation of recovery errors $\text{Corr}(\theta_i - \hat{\theta}_i, \theta_j - \hat{\theta}_j)$ to ensure that biases in $z$ and $v_b$ do not compensate for one another.
2. **Model Recovery Procedure (Rule 5):**
   - Simulate 50 synthetic datasets from each generative model (M1, M2, M3) using realistic empirical parameter distributions.
   - Fit all three models to each generated dataset (total 450 model fits).
   - Compute BIC for each fit and determine the winning model per dataset.
   - Construct the $3 \times 3$ **Confusion Matrix** $P(\text{fit } M_j \mid \text{sim } M_i)$ and the posterior **Inversion Matrix** $P(\text{sim } M_i \mid \text{fit } M_j)$, ensuring diagonal dominance ($> 85\%$ recovery accuracy).
3. **Posterior Predictive Validation (Rule 6):**
   - Use the best-fit parameters obtained from empirical mouse data to simulate synthetic sessions.
   - Overlay simulated and empirical psychometric curves, chronometric curves, and Conditional Bias Functions to confirm qualitative behavioral mimicry.

---

## 6. Evaluation Strategies & Behavioral Metrics

1. **Parameter Recovery Quality (Wilson & Collins Rule 4):**
   - Scatter plots of true vs. recovered parameters with unity ($y=x$) lines.
   - Summary table reporting recovery $r$, RMSE, and linear slope $\beta_1$ across trial counts ($N = 200, 500, 1000, 2000$).
   - Parameter error correlation heatmaps verifying absence of identifiability trade-offs.
2. **Model Identifiability (Wilson & Collins Rule 5):**
   - Confusion and inversion matrices with recovery rate percentages.
3. **Baseline Psychometric and Chronometric Reproduction (Part A):**
   - Psychometric curves: 4-parameter sigmoid fit across signed contrasts.
   - Chronometric curves: mean and median RT for correct and error trials across contrasts.
   - SAT demonstration: RT distribution widening and accuracy shifts across boundary heights $a$.
4. **Prior Mechanism Adjudication (Contribution 1):**
   - **Quantitative Model Selection:** $\Delta\text{BIC} = \text{BIC}_m - \text{BIC}_{\min}$ computed per mouse and summed across all 10 mice.
   - **Diagnostic Conditional Bias Function (CBF):** Compute empirical choice bias toward the 80% side across five RT quantiles (quintiles). Overlay the theoretical predictions from M1 (downward slope), M2 (flat line), and M3 (upward slope) against mouse data to inspect qualitative fit.
5. **Collapsing Boundary Assessment & Invariance (Contribution 2):**
   - Compare fixed vs. Weibull-collapsing bounds (with $s_v, s_z$ controls) via $\Delta\text{BIC}$.
   - Analyze 0%-contrast RT distribution tails under both 2.5 s and 5.0 s cutoffs.
   - Cross-mechanism validation: verify that the winning prior mechanism in C1 remains identical whether modeled under flat or collapsing boundaries.

---

## 7. Timeline & Work Distribution (2 people, 4 weeks)

| Week | Parth Dhodapkar | Joshua Koilpillai | Key Milestones |
|---|---|---|---|
| **Week 1** | Implement vectorized Euler–Maruyama DDM simulator with switches for M1, M2, M3, and collapsing bounds; verify numerical convergence | Download IBL biased-block dataset via ONE API; filter trials, extract RT, stimulus contrasts, block prior, and 1-back choice labels | Operational simulator; clean, verified IBL mouse dataset |
| **Week 2** | Implement Wilson & Collins Parameter Recovery pipeline; run synthetic parameter sweeps ($N=200, 500, 1000, 2000$); compute recovery stats ($r$, RMSE, error correlations) | Implement QMP Nelder-Mead fitting pipeline; run Model Recovery confusion matrix across M1–M3; fit baseline psychometric/chronometric curves (Part A) | Complete parameter recovery benchmarks; validated model recovery confusion matrix |
| **Week 3** | **Contribution 1 (C1):** Fit M1–M3 to 10 IBL mice; compute individual & group BIC; extract and plot Conditional Bias Functions | **Contribution 2 (C2):** Fit fixed vs. collapsing boundaries with $s_v, s_z$ controls; test robustness across 2.5 s and 5.0 s RT cutoffs; test C1 invariance | Quantitative model comparison results; empirical CBF curves; boundary collapse adjudication |
| **Week 4** | Posterior predictive checks (Rule 6); generate final publication-grade figures (recovery scatters, CBF fits, RT quantiles) | Draft manuscript sections on Novelty, Wilson & Collins recovery findings, and mechanistic interpretation; code repository clean-up | Final project report and code repository completed |

**Contingency Plan:** Contribution 1 (adjudicating the inferred prior mechanism) and the Wilson & Collins parameter recovery investigation represent the core scientific Deliverables. If fitting the collapsing boundary with trial variabilities (C2) encounters computational bottlenecks, C2 will be reported as an exploratory robustness subsection, fully protecting the completion and depth of C1 and the recovery analysis.

---

## 8. Key References

1. **Ashwood, Z. C., et al.** (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience*, 25(2), 201–212.
2. **Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D.** (2006). The physics of optimal decision making: A formal analysis of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4), 700–765.
3. **Chakravarty, S., et al.** (2024). A cross-species framework for investigating perceptual evidence accumulation. *bioRxiv*, 10.1101/2024.04.17.589945.
4. **Cisek, P., Puskas, G. A., & El-Murr, S.** (2009). Decisions in changing conditions: The urgency-gating model. *Journal of Neuroscience*, 29(37), 11560–11571.
5. **Drugowitsch, J., Moreno-Bote, R., Churchland, A. K., Shadlen, M. N., & Pouget, A.** (2012). The cost of accumulating evidence in perceptual decision making. *Journal of Neuroscience*, 32(11), 3612–3628.
6. **Fernandez, K., Fengler, A., & Urai, A. E.** (2025). Choice history biases evidence accumulation: A cross-species comparison from humans to mice. *bioRxiv*, 10.1101/2025.08.22.671879.
7. **Findling, C., et al.** (2023). Brain-wide representations of prior information in mouse decision-making. *bioRxiv*, 10.1101/2023.07.04.547684.
8. **Hanks, T. D., Mazurek, M. E., Kiani, R., Hopp, E., & Shadlen, M. N.** (2011). Elapsed decision time affects the weighting of prior probability in a perceptual decision task. *Journal of Neuroscience*, 31(17), 6339–6352.
9. **Hawkins, G. E., Forstmann, B. U., Wagenmakers, E. J., Ratcliff, R., & Brown, S. D.** (2015). Revisiting the evidence for collapsing boundaries and urgency signals in perceptual decision-making. *Journal of Neuroscience*, 35(6), 2476–2484.
10. **Heathcote, A., Brown, S., & Mewhort, D. J. K.** (2002). Quantile maximum likelihood estimation of response time distributions. *Psychonomic Bulletin & Review*, 9(2), 394–401.
11. **Hellmann, S., & Forster, C.** (2026). Two paths to bias: Distinct influences of stimulus probability and previous choice on drift diffusion parameters. *Computational Brain & Behavior*, 10.1007/s42113-026-00331-6.
12. **International Brain Laboratory.** (2021). Standardized and reproducible measurement of decision-making in mice. *eLife*, 10, e63711.
13. **Mulder, M. J., Wagenmakers, E. J., Ratcliff, R., Boekel, W., & Forstmann, B. U.** (2012). Bias in the brain: A diffusion model analysis of prior probability and potential payoff. *Journal of Neuroscience*, 32(7), 2335–2343.
14. **Ratcliff, R.** (1978). A theory of memory retrieval. *Psychological Review*, 85(2), 59–108.
15. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: Theory and data for two-choice decisions. *Neural Computation*, 20(4), 873–922.
16. **Ratcliff, R., & Tuerlinckx, F.** (2002). Estimating parameters of the diffusion model: Approaches to dealing with contaminant trials and parameter estimates. *Psychonomic Bulletin & Review*, 9(3), 438–481.
17. **Summerfield, C., & de Lange, F. P.** (2014). Expectation in perceptual decision making: Neural and computational mechanisms. *Nature Reviews Neuroscience*, 15(11), 745–756.
18. **Urai, A. E., de Gee, J. W., Tsetsos, K., & Donner, T. H.** (2019). Choice history biases subsequent evidence accumulation. *eLife*, 8, e46331.
19. **White, C. N., & Poldrack, R. A.** (2014). Decomposing bias in different types of simple decisions. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 40(2), 385–398.
20. **Wilson, R. C., & Collins, A. G.** (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*, 8, e49547. https://doi.org/10.7554/eLife.49547
