# Project Proposal: Modeling Simple Decisions and the Speed-Accuracy Trade-off Using the Drift Diffusion Model

**Authors:** Parth Dhodapkar (2024111009), Joshua Koilpillai (2024101128)  
**Course:** Introduction to Neural and Cognitive Modeling (INCM) | **Topic:** Evidence Accumulation (Question 27)

---

## 1. Project Scope & Research Questions

When deciding between two options, such as whether a faint target appeared on the left or right of a screen, the brain does not guess on a single snapshot. Instead, it accumulates sensory evidence over a few hundred milliseconds until reaching a decision threshold. This leads to two familiar patterns:
1. **Easy tasks:** Clear stimuli trigger fast decisions with very few mistakes.
2. **Hard tasks:** Faint or ambiguous stimuli take longer to resolve and result in more errors.

This project investigates how evidence accumulation explains both **what people or animals choose** and **how long they take to respond**, and how changing the decision threshold explains the **speed-accuracy trade-off (SAT)**.

### What We Plan to Do:
- **Build a simulation in Python:** Write a trial-by-trial simulation of the Drift Diffusion Model (DDM) using `numpy` and `matplotlib`.
- **Vary task difficulty:** Test the model across different stimulus contrasts (from 0% up to 100%) to reproduce psychometric (accuracy) and chronometric (response time) curves.
- **Analyze Speed vs. Accuracy:** Vary decision thresholds from low (rushed) to high (cautious) to map out how caution changes reaction times and error rates. We also implement a collapsing boundary to see if time-decaying thresholds better capture how subjects terminate difficult trials.
- **Evaluate Model Assumptions:** Examine the DDM's core assumptions (constant caution, continuous integration, and independent motor delays) and assess where they make sense or fall short biologically.

### Core Questions:
- How well does noisy evidence accumulation reproduce empirical accuracy and reaction time curves across task difficulties?
- How does adjusting the decision boundary explain the speed-accuracy trade-off, and can collapsing boundaries resolve the unrealistically long reaction times predicted on zero-contrast trials?
- What do the model's assumptions imply about how the brain processes uncertainty and deliberation costs?

---

## 2. Literature Review

The Drift Diffusion Model is the standard computational tool for studying two-alternative forced-choice (2AFC) decisions:
- **Foundational DDM:** Ratcliff (1978) and Ratcliff & McKoon (2008) demonstrated that modeling decisions as a continuous random walk with drift naturally explains right-skewed reaction time distributions and why error trials often have different latencies than correct trials.
- **Mathematical Optimality:** Bogacz et al. (2006) proved that under constant noise and flat boundaries, the DDM is the continuous-time version of the Sequential Probability Ratio Test (SPRT). For any fixed error rate, it is the mathematically fastest way to make a decision.
- **Urgency & Collapsing Boundaries:** On difficult or zero-evidence trials, standard flat boundaries predict excessively long reaction times. Drugowitsch et al. (2012) and Cisek et al. (2009) showed that biological agents use dynamic urgency by lowering decision thresholds over time to avoid wasting time on impossible choices and maximize reward rates.
- **Empirical Benchmarks:** Gold & Shadlen (2007) linked evidence accumulation to ramping neural activity in parietal cortex, while the International Brain Laboratory (IBL, 2021) provided standardized, large-scale behavioral datasets of visual 2AFC decisions in mice.

<div style="page-break-after: always;"></div>

## 3. Planned Datasets

We will evaluate our simulations using two data sources:
1. **IBL Mouse Visual Decision Dataset:** Publicly available behavioral data from the International Brain Laboratory (2021). Head-fixed mice turn a steering wheel to report whether a visual grating appeared on the left or right across 6 calibrated contrast levels ($0\%$, $6.25\%$, $12.5\%$, $25\%$, $50\%$, and $100\%$). We will extract trial contrasts, choice outcomes, and reaction times, filtering out anticipatory responses ($< 100\,\text{ms}$) and disengaged trials ($> 2.5\,\text{s}$).
2. **Synthetic Benchmark Data:** Simulated trials generated across known parameter ranges (drift rates $v$, thresholds $a$, and non-decision times $t_0$) to verify parameter recovery and test our simulation pipeline before fitting empirical data.

---

## 4. Methodologies

### 4.1 The Model
Deliberation is modeled as a 1D evidence counter $x(t)$ starting at neutral ($x(0) = 0$):
$$x(t + \Delta t) = x(t) + v(c)\cdot\Delta t + \sigma\sqrt{\Delta t}\cdot\xi(t), \quad \xi(t) \sim \mathcal{N}(0, 1)$$
- **Drift Rate ($v$):** Scales with stimulus contrast: $v(c) = k \cdot c$, where $k$ is sensory sensitivity.
- **Noise ($\sigma$):** Neural variability, modeled with standard deviation $\sigma = 1$.
- **Boundaries ($\pm a$):** Deliberation terminates when $x(t)$ hits $+a$ (choose Left) or $-a$ (choose Right), yielding decision time $T_d$.
- **Collapsing Boundary Extension:** We also test an urgency-based collapsing boundary, $\theta(t) = \pm a \cdot \exp(-(t/\lambda)^\gamma)$, where thresholds decay over trial time.

Observed reaction time adds non-decision sensory and motor execution latencies ($t_0$):
$$\text{RT} = T_d + t_0 \quad (t_0 \approx 0.15\text{--}0.3\,\text{s})$$

### 4.2 Numerical Simulation & Fitting
We simulate individual trials via Euler-Maruyama numerical integration ($\Delta t = 1\,\text{ms}$, maximum trial duration $3\,\text{s}$). Parameters ($k, a, t_0$) will be calibrated to empirical session data by minimizing prediction error on choice fractions and reaction time distributions using Nelder-Mead simplex optimization.

---

## 5. Evaluation Strategies

We will evaluate the model using five key strategies:
1. **Psychometric Curves:** Plot choice accuracy against contrast levels to see how well the model reproduces sensory thresholds and error rates.
2. **Chronometric Curves:** Compare mean reaction times for correct and error trials across contrasts to evaluate whether deliberation speed scales appropriately with task difficulty.
3. **RT Distributions:** Compare simulated and empirical reaction time histograms to verify that the model reproduces the characteristic right-skewed shape.
4. **Speed vs. Accuracy Trade-off:** Systematically vary boundary height $a$ to map the trade-off curve between decision speed and error rate.
5. **Model Comparison:** Compare the fixed-boundary and collapsing-boundary models on low-contrast trials using likelihood / BIC to check whether adding urgency meaningfully improves fits to animal behavior.

---

## 6. Key References

1. **Bogacz, R., et al.** (2006). The physics of optimal decision making. *Psychological Review*, 113(4), 700–765.
2. **Cisek, P., et al.** (2009). Dynamic urgency overrides sensory accumulation. *Journal of Neuroscience*, 29(37), 11560–11573.
3. **Drugowitsch, J., et al.** (2012). The cost of accumulating evidence in perceptual decision making. *Journal of Neuroscience*, 32(11), 3612–3628.
4. **International Brain Laboratory.** (2021). Standardized decision-making in mice. *eLife*, 10, e63711.
5. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: theory and data for two-choice decisions. *Neural Computation*, 20(4), 873–922.