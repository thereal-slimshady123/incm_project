# Project Proposal: Modeling Simple Decisions and the Speed-Accuracy Trade-off Using the Drift Diffusion Model
- Parth Dhodapkar 2024111009
- Joshua Koilpillai 2024101128

## 1. Introduction: What is this project about?

When you have to make a quick choice between two options, for example, deciding whether a faint image appeared on the **left** vs **right** of a screen, your brain gathers visual information over a few hundred milliseconds until it feels confident enough to make a choice.

This leads to two common-sense observations:
1. **Easy tasks:** If the image is bright and obvious, you decide very quickly and rarely make mistakes.
2. **Hard tasks:** If the image is faint or blurry, you take longer to decide and make more mistakes.

The classic computational model used to explain this behavior is the **Drift Diffusion Model (DDM)**. It treats decision-making like an evidence counter:
- You start at neutral ($0$).
- As you look at the screen, sensory evidence pushes your internal counter toward the "left" or "right" threshold
- Because neural signals are noisy, the counter doesn't move in a straight line and instead it moves up and down randomly as it drifts.
- The moment the counter hits either threshold, you commit to that choice and press the button.

In this project, we will build a simulation of this model in Python to study how evidence accumulation explains both **what people/animals choose** and **how long they take to respond**.

## 2. Research Question*

Our specific question is: "**How does evidence accumulation explain both choice accuracy and response times across easy and difficult tasks? Furthermore, how does adjusting the decision threshold explain the "speed vs. accuracy" trade-off, and what does this assume about how the brain makes decisions?**"

### What we plan to do:
1. **Simulate the model in Python:** Write a simple script that simulates trial-by-trial decisions.
2. **Vary task difficulty:** Test whether the model reproduces the expected patterns across different stimulus contrast levels (from 0% contrast up to 100% contrast).
3. **Test the Speed-Accuracy Trade-off:** Change the threshold height to see how being "cautious" (high threshold) versus "in a rush" (low threshold) changes accuracy and reaction time.
4. **Examine the model's assumptions:** Discuss what the model assumes about how the brain processes information and where those assumptions make sense or fall short.

## 3. How the Model Works

The DDM models decision-making as a continuous, noisy accumulation of evidence $x(t)$ starting at neutral ($x(0) = 0$). At each discrete time step $\Delta t$, evidence updates as:

$$x(t + \Delta t) = x(t) + v \cdot \Delta t + \text{noise}$$

- **Drift Rate ($v$):** Reflects stimulus strength/clarity. Higher contrast produces larger $v$ (faster drift toward target); faint or zero-contrast stimuli yield slow or zero drift ($v \approx 0$).
- **Noise:** Random neural fluctuations modeled as Gaussian noise ($\sigma \sqrt{\Delta t}$).
- **Boundaries ($\pm a$):** Deliberation terminates when $x(t)$ crosses $+a$ (choose Left) or $-a$ (choose Right), yielding the **decision time** ($T_d$).

The total **reaction time (RT)** incorporates physical delays via non-decision time ($t_0 \approx 0.2\,\text{s}$), covering both sensory encoding and motor execution (time taken to act):

$$\text{RT} = T_d + t_0$$

## 4. Model Assumptions

1. The brain accumulates evidence rather than guessing on snapshots. 
2. Caution is controlled by boundary separation ($a$).
3. Caution stays constant throughout the trial.
4. Deliberation and motor execution are separate.


## 5. Key References

1. **Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D.** (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4), 700–765.
2. **International Brain Laboratory.** (2021). A standardized and reproducible method to measure decision-making in mice. *eLife*, 10, e63711.
3. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: theory and data for two-choice decisions. *Neural Computation*, 20(4), 873–922.