# Project Proposal: How Does an Uncued, Inferred Prior Enter the Decision Process? A Drift Diffusion Model Study of Mouse Perceptual Choice

**Authors:** Parth Dhodapkar (2024111009), Joshua Koilpillai (2024101128)  
**Course:** Introduction to Neural and Cognitive Modeling (INCM) | **Topic:** Evidence Accumulation (Question 27)

---

## 1. Project Scope & Research Questions

When deciding between two options, such as whether a faint target appeared on the left or right of a screen, the brain does not guess from a single snapshot. It accumulates noisy sensory evidence over a few hundred milliseconds until it reaches a decision threshold. Easy stimuli produce fast, accurate choices; faint stimuli produce slow choices and more errors. The Drift Diffusion Model (DDM) captures both patterns, and changing its threshold explains the **speed-accuracy trade-off (SAT)**.

Decisions are also shaped by what we expect before the stimulus appears. In the DDM, an expectation can enter in different ways: by **moving the starting point** of accumulation, by **biasing the drift** so that evidence is read in favour of the expected side, or as a **bias that grows the longer the decision takes**. Recent work suggests that the route depends on the *type* of expectation:
- **Explicit, cued priors** (e.g., "the next stimulus is 80% likely to be left") shift the **starting point** (Mulder et al., 2012; Hellmann & Forster, 2026).
- **Implicit biases from the previous choice** shift the **drift** (Urai et al., 2019; Hellmann & Forster, 2026), and this also holds for IBL mice (Fernandez et al., 2025).

**The gap.** In the International Brain Laboratory (IBL) task, the stimulus appears on one side 80% of the time for a block of trials, then switches, and mice are **never told** which block they are in. They must infer the prior from experience, and they do (Findling et al.). This prior is neither cued nor a simple repeat of the previous choice, so the existing work does not predict which route it takes. No study we found has fit a DDM to how this inferred prior enters mouse decisions: Fernandez et al. (2025) fit DDMs to IBL mice but deliberately **excluded** the sessions with probability blocks.

**The project is purely computational.** We collect no new data. Every analysis is a simulation or a model fit to the existing public IBL dataset, and each question is answered by comparing how well competing model variants fit the same recorded trials.

### Part A: Reproduction (baseline)
- Build a trial-by-trial DDM simulator in Python (`numpy`, `scipy`, `matplotlib`), and verify it by recovering known parameters from synthetic data.
- Reproduce psychometric (accuracy) and chronometric (RT) curves across stimulus contrasts, and simulate the SAT by sweeping the boundary height $a$.

### Part B: Contributions
**C1 (main). Which route does the inferred block prior take?** We fit three competing mechanisms, each with a previous-choice drift term so that a block effect cannot be mistaken for choice repetition:

| Mechanism | How the prior enters | Predicted bias toward the likely side vs. RT |
|---|---|---|
| **M1: Starting point** | $x(0) = z \cdot \text{prior} \cdot a$ | strongest for **fast** responses, then **declines** |
| **M2: Drift bias** | $v \to v + v_b \cdot \text{prior}$ | roughly **flat** across RTs |
| **M3: Dynamic bias** (Hanks et al., 2011) | $x(t) + \text{prior} \cdot b_{\max}\, \frac{t}{t + \tau_b}$ | **increases** for slow responses |

The three mechanisms predict three different shapes of the conditional bias function (White & Poldrack, 2014), so the result can be read from one figure, alongside the BIC comparison. Any outcome is informative:
- **M1** would mean an inferred prior behaves like an explicit one.
- **M2** would mean it behaves like choice history.
- **M3** would mean mice weight the prior more as the evidence fails to settle the decision, as Hanks et al. found in monkeys.

**C2 (secondary). Is urgency needed on the IBL wheel task?** Collapsing boundaries have been found in mice on a nose-poke task (Chakravarty et al., 2024) but not tested on the IBL wheel task. We compare a fixed and a collapsing boundary, **both** including trial-to-trial drift variability ($s_v$) and starting-point variability ($s_z$), which can otherwise mimic urgency. This matters for C1 as well: a collapsing boundary and a dynamic bias both act mainly on slow trials, so we check that the C1 result holds under the winning boundary.

### Core Questions
1. How well does noisy evidence accumulation reproduce mouse accuracy and RT across contrasts? *(Part A)*
2. Does the uncued, inferred IBL block prior act like an explicit prior (starting point), like choice history (drift), or as a time-growing bias? *(C1)*
3. Do collapsing boundaries beat fixed boundaries on IBL wheel responses once trial-to-trial variability is included? *(C2)*

---

## 2. Novelty Statement

| Closest prior work | What it did | What it left open | What we add |
|---|---|---|---|
| Fernandez, Fengler & Urai (2025) | DDM on IBL mice; previous choice → drift | Excluded biased-block sessions | DDM on the **biased** sessions |
| Hellmann & Forster (2026); Mulder et al. (2012) | Explicit cued prior → starting point (humans) | Priors that are inferred, not cued | Test an **uncued, inferred** prior |
| Findling et al. | Mice infer the block prior; brain-wide representation | No decision-process (DDM) model | Mechanism-level DDM account |
| Hanks et al. (2011) | Time-growing prior bias (monkeys, humans) | Not tested in rodents | Include it as a competing mechanism in mice |
| Chakravarty et al. (2024) | Collapsing bounds in mice, nose-poke task | Wheel task untested | Fixed vs. collapsing on IBL with $s_v, s_z$ controls |

Our methods (DDM, quantile fitting, BIC) are standard. The contribution is the question: **where an inferred prior enters the decision process**, tested on a large public dataset where it has not been modelled.

<div style="page-break-after: always;"></div>

## 3. Literature Review

- **Foundational DDM:** Ratcliff (1978) and Ratcliff & McKoon (2008) showed that a noisy random walk to a boundary explains right-skewed RT distributions and error RTs. Ratcliff & Tuerlinckx (2002) showed that trial-to-trial variability in drift ($s_v$) and starting point ($s_z$) produces slow and fast errors respectively. Bogacz et al. (2006) showed that the flat-boundary DDM implements the optimal Sequential Probability Ratio Test.
- **Prior and history bias:** Cued priors mainly shift the starting point (Mulder et al., 2012; Hellmann & Forster, 2026), although some studies find accumulation-rate effects. Choice history shifts the drift (Urai et al., 2019), including in IBL mice (Fernandez et al., 2025). Hanks et al. (2011) found that prior information acts as a bias signal that grows with elapsed decision time. White & Poldrack (2014) introduced the conditional bias function for telling these mechanisms apart.
- **Urgency:** Cisek et al. (2009) and Drugowitsch et al. (2012) argued that agents should lower their thresholds over time when deliberation is costly. Hawkins et al. (2015) found that fixed boundaries usually fit humans and monkeys better, while Chakravarty et al. (2024) found collapsing boundaries in rats and mice.
- **Mouse behaviour:** IBL mice infer the block prior (Findling et al.) and switch between engaged and biased strategies within a session (Ashwood et al., 2022).

---

## 4. Planned Datasets

1. **IBL Mouse Visual Decision Dataset** (IBL, 2021): already recorded and freely downloadable through the public ONE API; we run no experiments ourselves. Head-fixed mice turn a wheel to report whether a grating appeared on the left or right at contrasts of $0, 6.25, 12.5, 25, 50, 100\%$. We use ~10 trained mice from the *biased* protocol (20/50/80 probability blocks). For each trial we extract the signed contrast, block probability, previous choice, choice, and **RT = first wheel movement − stimulus onset**. We exclude anticipatory ($< 100\,\text{ms}$) and disengaged ($> 2.5\,\text{s}$) trials, and rerun C2 with a $5\,\text{s}$ cutoff because the long-RT tail is where fixed and collapsing boundaries differ.
2. **Synthetic Benchmark Data:** Trials simulated from each mechanism with known parameters, used to confirm that the model comparison recovers the true mechanism at our trial counts before we fit real data.

---

## 5. Methodologies

### 5.1 One Model Family
All models share one evidence counter $x(t)$:
$$x(t + \Delta t) = x(t) + v\,\Delta t + \sigma\sqrt{\Delta t}\,\xi(t), \quad \xi(t) \sim \mathcal{N}(0, 1),\ \sigma = 1$$
with drift $v = k\,\text{sign}(c)\,\lvert c\rvert^{\beta} + v_h \cdot \text{prev. choice}$, boundaries at $\pm a$, and RT $= T_d + t_0$ (upper boundary = right choice). Here *prior* $\in \{-1, 0, +1\}$ codes the 20/50/80 block and *prev. choice* $\in \{-1, +1\}$. M1–M3 add the prior term from the table in §1. C2 adds $s_v$ and $s_z$, and replaces $\pm a$ with the Weibull collapse $\pm a\,\exp(-(t/\lambda)^\gamma)$ (Hawkins et al., 2015).

### 5.2 Simulation & Fitting
Trials are simulated in vectorized form with Euler–Maruyama integration ($\Delta t = 1\,\text{ms}$, cut off at $5\,\text{s}$). We fit all models in the same way, since the collapsing and dynamic-bias models have no simple closed-form likelihood. For each contrast and choice, we compare simulated and observed RT quantiles ($0.1, 0.3, 0.5, 0.7, 0.9$) using the quantile maximum probability method (Heathcote et al., 2002), which gives an approximate likelihood for BIC. Parameters are fitted per mouse with Nelder–Mead, using the same random noise sequences on every evaluation to keep the objective smooth.

---

## 6. Evaluation Strategies

1. **Model recovery** *(A)*: simulate each mechanism, refit all three, and report a confusion matrix showing whether the comparison identifies the true mechanism at our trial counts.
2. **Psychometric & chronometric fits** *(A)*: model vs. mouse accuracy and mean correct/error RT for each contrast.
3. **Prior mechanism** *(C1, key result)*: BIC for M1–M3, summed across mice and counted per mouse, plus the conditional bias function from the data overlaid with each model's prediction (declining vs. flat vs. rising).
4. **Boundary shape** *(C2)*: BIC for fixed vs. collapsing (both with $s_v, s_z$), the 0%-contrast RT distribution in 50/50 blocks, a rerun at the $5\,\text{s}$ cutoff, and a check that the C1 winner is unchanged under the C2 winner.

---

## 7. Timeline (2 people, 4 weeks)

| Week | Person 1 | Person 2 |
|---|---|---|
| 1 | Simulator with all switches | IBL download, filtering, RT, block and history labels |
| 2 | QMP fitting; model recovery for M1–M3 | Baseline fits; psychometric/chronometric plots *(Part A)* |
| 3 | **C1**: fit M1–M3; conditional bias functions | **C2**: fixed vs. collapsing |
| 4 | Report and figures (shared) | Report and figures (shared) |

**If time runs short:** C1 is the contribution and is protected; C2 can be reduced to a short robustness section.

---

## 8. Key References

1. **Ashwood, Z. C., et al.** (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience*, 25(2), 201–212.
2. **Bogacz, R., et al.** (2006). The physics of optimal decision making. *Psychological Review*, 113(4), 700–765.
3. **Chakravarty, S., et al.** (2024). A cross-species framework for investigating perceptual evidence accumulation. *bioRxiv*, 10.1101/2024.04.17.589945.
4. **Cisek, P., et al.** (2009). Decisions in changing conditions: The urgency-gating model. *Journal of Neuroscience*, 29(37), 11560–11571.
5. **Drugowitsch, J., et al.** (2012). The cost of accumulating evidence in perceptual decision making. *Journal of Neuroscience*, 32(11), 3612–3628.
6. **Fernandez, K., Fengler, A., & Urai, A. E.** (2025). Choice history biases evidence accumulation: A cross-species comparison from humans to mice. *bioRxiv*, 10.1101/2025.08.22.671879.
7. **Findling, C., et al.** Brain-wide representations of prior information in mouse decision-making. *bioRxiv*, 10.1101/2023.07.04.547684.
8. **Hanks, T. D., Mazurek, M. E., Kiani, R., Hopp, E., & Shadlen, M. N.** (2011). Elapsed decision time affects the weighting of prior probability in a perceptual decision task. *Journal of Neuroscience*, 31(17), 6339–6352.
9. **Hawkins, G. E., et al.** (2015). Revisiting the evidence for collapsing boundaries and urgency signals in perceptual decision-making. *Journal of Neuroscience*, 35(6), 2476–2484.
10. **Heathcote, A., Brown, S., & Mewhort, D. J. K.** (2002). Quantile maximum likelihood estimation of response time distributions. *Psychonomic Bulletin & Review*, 9(2), 394–401.
11. **Hellmann, & Forster, C.** (2026). Two paths to bias: Distinct influences of stimulus probability and previous choice on drift diffusion parameters. *Computational Brain & Behavior*, 10.1007/s42113-026-00331-6.
12. **International Brain Laboratory.** (2021). Standardized and reproducible measurement of decision-making in mice. *eLife*, 10, e63711.
13. **Mulder, M. J., et al.** (2012). Bias in the brain: A diffusion model analysis of prior probability and potential payoff. *Journal of Neuroscience*, 32(7), 2335–2343.
14. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: Theory and data for two-choice decisions. *Neural Computation*, 20(4), 873–922.
15. **Ratcliff, R., & Tuerlinckx, F.** (2002). Estimating parameters of the diffusion model. *Psychonomic Bulletin & Review*, 9(3), 438–481.
16. **Urai, A. E., et al.** (2019). Choice history biases subsequent evidence accumulation. *eLife*, 8, e46331.
17. **White, C. N., & Poldrack, R. A.** (2014). Decomposing bias in different types of simple decisions. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 40(2), 385–398.
