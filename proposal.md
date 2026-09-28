# Project Proposal: Modeling Simple Decisions and the Speed-Accuracy Trade-off Using the Drift Diffusion Model

**Course:** Introduction to Neural and Cognitive Modeling (INCM)  
**Topic:** Question 27 – Evidence Accumulation in Decision-Making  

---

## 1. Introduction: What is this project about?

When you have to make a quick choice between two options—for example, deciding whether a faint image appeared on the **Left** or the **Right** of a screen, your brain doesn't guess instantly. Instead, it gathers visual information over a few hundred milliseconds until it feels confident enough to make a choice.

This leads to two common-sense observations:
1. **Easy tasks:** If the image is bright and obvious, you decide very quickly and rarely make mistakes.
2. **Hard tasks:** If the image is faint or blurry, you take longer to decide and make more mistakes.

The classic computational model used to explain this behavior is the **Drift Diffusion Model (DDM)**. It treats decision-making like an evidence counter:
- You start at neutral ($0$).
- As you look at the screen, sensory evidence pushes your internal counter toward the "Left" threshold or the "Right" threshold.
- Because neural signals are noisy, the counter doesn't move in a straight line and instead it moves up and down randomly as it drifts.
- The moment the counter hits either threshold, you commit to that choice and press the button.

In this project, we will build a simulation of this model in Python to study how evidence accumulation explains both **what people/animals choose** and **how long they take to respond**.

---

## 2. Research Question

We are addressing Question 27 from the projects list:
> *"Explore how the accumulation of evidence can account for both choices and response times. Develop a question using a drift diffusion model... Consider what the model's assumptions imply about the decision process."*

Our specific question is:

> **How does evidence accumulation explain both choice accuracy and response times across easy and difficult tasks? Furthermore, how does adjusting the decision threshold explain the "speed vs. accuracy" trade-off, and what does this assume about how the brain makes decisions?**

### What we plan to do:
1. **Simulate the model in Python:** Write a simple script that simulates trial-by-trial decisions.
2. **Vary task difficulty:** Test whether the model reproduces the expected patterns across different stimulus contrast levels (from 0% contrast up to 100% contrast).
3. **Test the Speed-Accuracy Trade-off:** Change the threshold height to see how being "cautious" (high threshold) versus "in a rush" (low threshold) changes accuracy and reaction time.
4. **Examine the model's assumptions:** Discuss what the model assumes about how the brain processes information and where those assumptions make sense or fall short.

---

## 3. How the Model Works (The Math Made Simple)

The model tracks an internal evidence score, which we call $x(t)$.

### 3.1 The Evidence Accumulation Rule
At the start of a trial ($t = 0$), the score starts at zero: $x(0) = 0$ (neutral).

At each tiny time step (for example, every $1\,\text{millisecond}$), the evidence score updates using this simple rule:

$$\text{New Score} = \text{Old Score} + (\text{Signal Drift}) + (\text{Random Noise})$$

In mathematical form:
$$x(t + \Delta t) = x(t) + v \cdot \Delta t + \text{noise}$$

Where:
- **$v$ (Drift Rate):** How strong and clear the sensory evidence is.
  - If the stimulus is bright and clearly on the Left, $v$ is positive and large, pulling the score rapidly upward.
  - If the stimulus is faint, $v$ is small, so the score drifts slowly.
  - If there is no stimulus at all (0% contrast), $v = 0$, meaning the score just wanders randomly due to noise.
- **$\text{noise}$:** Random neural fluctuations. Even when looking at a still image, neurons in the visual cortex fire with variability.
- **$\Delta t$:** The time step of the simulation (e.g., $0.001\,\text{s}$).

### 3.2 Decision Boundaries and Reaction Time
We define two boundaries:
- **Upper boundary ($+a$):** Choose **Left**
- **Lower boundary ($-a$):** Choose **Right**

As soon as the score reaches $+a$ or $-a$:
1. Deliberation stops. The time taken to reach the boundary is the **decision time** ($T_d$).
2. The total **reaction time (RT)** is the decision time plus a small physical delay ($t_0$):
   $$\text{RT} = T_d + t_0$$
   Here, $t_0$ (non-decision time, usually $\approx 0.2\,\text{seconds}$) accounts for the physical time it takes for light to hit the retina and for the hand or paw to physically execute the movement.

---

## 4. What Do the Model's Assumptions Imply About the Brain?

Question 27 explicitly asks us to consider what the model's assumptions imply about the decision process. Here is what the model assumes, translated into plain terms:

1. **The brain accumulates evidence rather than guessing on snapshots:**  
   The model assumes that the brain does not make a choice from a single glance. Instead, it adds up sensory inputs over time to average out noise, which improves decision quality.
2. **Caution is controlled by boundary separation ($a$):**  
   The model assumes that when you decide to be careful, you don't think "faster" or "slower"—you simply raise your internal threshold ($a$). 
   - **Low threshold ($a$):** You need very little evidence to jump to a conclusion. You react quickly, but random noise can easily trigger mistakes (Speed mode).
   - **High threshold ($a$):** You wait until you have overwhelming evidence. You make very few mistakes, but it takes longer (Accuracy mode).
3. **Caution stays constant throughout the trial:**  
   The standard DDM assumes the boundaries $+a$ and $-a$ remain flat and unchanging over time. This implies that the brain remains just as patient after 2 seconds of staring at a blank screen as it was at the very start.
4. **Deliberation and motor execution are separate:**  
   The model assumes that deciding which button to press ($T_d$) and the physical act of pressing it ($t_0$) are completely independent steps that simply add together.

---

## 5. Implementation Plan

We will implement the simulation in Python using standard libraries (`numpy` and `matplotlib`).

### Step 1: Trial Simulation
We will simulate thousands of individual decision trials by running the step-by-step update rule until the score hits $+a$ or $-a$.

### Step 2: Varying Difficulty (Evidence Strength)
We will test 6 contrast levels matching real experiment settings (such as the International Brain Laboratory mouse visual task):
- $0\%$ contrast (impossible / pure guess)
- $6.25\%$ contrast (very hard)
- $12.5\%$ contrast (hard)
- $25\%$ contrast (medium)
- $50\%$ contrast (easy)
- $100\%$ contrast (very easy)

For each contrast level, we will record:
- What fraction of trials were correct (**Accuracy**).
- How long each trial took (**Mean Reaction Time**).

### Step 3: Varying Caution (Speed vs. Accuracy)
We will run the simulations under three threshold settings:
- **Low threshold ($a = 0.6$):** Simulating a subject rushing to answer quickly.
- **Medium threshold ($a = 1.0$):** Standard baseline performance.
- **High threshold ($a = 1.6$):** Simulating a subject being extra careful.

---

## 6. Expected Outcomes & Visualizations

Through these simulations, we aim to examine and visualize:

1. **Sample Decision Trajectories:** Visualizing a few individual evidence paths over time to see how sensory noise and drift influence how trials reach a boundary.
2. **Accuracy Trends across Contrasts (Psychometric Curve):** Plotting choice accuracy against stimulus contrast to observe how decision quality changes as evidence becomes clearer or more ambiguous.
3. **Response Time Trends (Chronometric Curve):** Analyzing average response latencies across contrast levels to see how deliberation time varies with task difficulty.
4. **Threshold Comparisons (Speed vs. Accuracy):** Comparing performance curves under different boundary settings to observe how altering the threshold affects reaction times and error rates.
5. **Evaluation of Assumptions:** Discussing how well this simple accumulation mechanism captures typical decision patterns and noting where its basic assumptions might oversimplify biological decision-making.


---

## 7. Key References

1. **Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D.** (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. *Psychological Review*, 113(4), 700–765.
2. **International Brain Laboratory.** (2021). A standardized and reproducible method to measure decision-making in mice. *eLife*, 10, e63711.
3. **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: theory and data for two-choice decisions. *Neural Computation*, 20(4), 873–922.