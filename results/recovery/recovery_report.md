# Plan-scale recovery report (Wilson & Collins 2019, Rules 4 and 5)

## Model recovery (Rule 5): 150 datasets (1,000 trials each, the pipeline default)

BIC confusion matrix P(fitted | true), counts:

| true_model | M1 | M2 | M3 |
|---|---|---|---|
| M1 | 47 | 3 | 0 |
| M2 | 4 | 45 | 1 |
| M3 | 0 | 16 | 34 |

P(fitted | true):

| true_model | M1 | M2 | M3 |
|---|---|---|---|
| M1 | 94.0% | 6.0% | 0.0% |
| M2 | 8.0% | 90.0% | 2.0% |
| M3 | 0.0% | 32.0% | 68.0% |

Inversion P(true | fitted):

| true_model | M1 | M2 | M3 |
|---|---|---|---|
| M1 | 92.2% | 4.7% | 0.0% |
| M2 | 7.8% | 70.3% | 2.9% |
| M3 | 0.0% | 25.0% | 97.1% |

Recovery per mechanism: M1 94%, M2 90%, M3 68%; overall 84%. **Criterion (> 85% for every mechanism): NOT MET.**

Datasets won by less than 10 BIC: 33% overall (M1 14%, M2 24%, M3 60%).

Recovery by size of the generating prior effect (terciles of |z|, |v_b|, b_max):

| true_model | effect_range | n | recovered | mean_margin_bic |
|---|---|---|---|---|
| M1 | 0.02-0.15 | 17 | 0.82 | 23.74 |
| M1 | 0.15-0.35 | 16 | 1.00 | 127.30 |
| M1 | 0.35-0.50 | 17 | 1.00 | 419.21 |
| M2 | 0.00-0.36 | 17 | 0.82 | 15.58 |
| M2 | 0.36-0.79 | 16 | 0.94 | 124.34 |
| M2 | 0.79-1.20 | 17 | 0.94 | 254.68 |
| M3 | 0.20-0.69 | 17 | 0.35 | 6.10 |
| M3 | 0.69-1.05 | 16 | 0.75 | 9.67 |
| M3 | 1.05-1.48 | 17 | 0.94 | 20.68 |

## Parameter recovery (Rule 4)

Pearson r of recovered vs true parameter (criterion r > 0.8); `*` marks r <= 0.8.

### JOINT ({200: 100, 500: 100, 1000: 100, 2000: 100} parameter vectors per N)

Pearson r:

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.92 | 0.97 | 0.99 | 0.99 |
| k | 0.78* | 0.94 | 0.96 | 0.98 |
| t0 | 0.81 | 0.99 | 0.99 | 0.97 |
| v_b | 0.96 | 0.98 | 0.99 | 0.99 |
| v_h | 0.89 | 0.97 | 0.98 | 0.98 |
| z | 0.95 | 0.98 | 0.98 | 0.99 |

Slope of recovered on true (1 = unbiased):

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.86 | 0.96 | 0.98 | 1.00 |
| k | 0.81 | 0.92 | 0.96 | 0.98 |
| t0 | 0.96 | 1.00 | 1.01 | 0.97 |
| v_b | 0.94 | 0.93 | 0.95 | 0.96 |
| v_h | 0.86 | 1.00 | 0.96 | 0.97 |
| z | 0.99 | 1.02 | 1.01 | 1.06 |

Largest error correlation per N (cross-talk check, |r| > 0.3 flagged):

| n_trials | param_a | param_b | error_corr |
|---|---|---|---|
| 200 | z | v_b | -0.603 |
| 500 | z | v_b | -0.672 |
| 1000 | a | k | 0.669 |
| 2000 | z | v_b | -0.570 |

**z vs v_b error correlation (the plan's cross-talk test):** N=200: -0.60, N=500: -0.67, N=1000: -0.65, N=2000: -0.57

### M1 ({200: 100, 500: 100, 1000: 100, 2000: 100} parameter vectors per N)

Pearson r:

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.95 | 0.99 | 0.99 | 0.99 |
| k | 0.83 | 0.97 | 0.98 | 0.98 |
| t0 | 0.95 | 0.99 | 0.99 | 0.99 |
| v_h | 0.94 | 0.97 | 0.98 | 0.99 |
| z | 0.98 | 0.99 | 0.99 | 1.00 |

Slope of recovered on true (1 = unbiased):

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.96 | 0.97 | 0.98 | 1.02 |
| k | 0.90 | 0.96 | 0.95 | 1.01 |
| t0 | 1.04 | 0.99 | 0.99 | 1.01 |
| v_h | 0.95 | 0.95 | 0.95 | 0.95 |
| z | 1.04 | 1.02 | 0.99 | 1.02 |

Largest error correlation per N (cross-talk check, |r| > 0.3 flagged):

| n_trials | param_a | param_b | error_corr |
|---|---|---|---|
| 200 | a | k | 0.435 |
| 500 | a | k | 0.321 |
| 1000 | a | k | 0.372 |
| 2000 | a | k | 0.569 |

### M2 ({200: 100, 500: 100, 1000: 100, 2000: 100} parameter vectors per N)

Pearson r:

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.96 | 0.98 | 0.99 | 0.99 |
| k | 0.83 | 0.93 | 0.97 | 0.98 |
| t0 | 0.92 | 0.97 | 0.99 | 0.99 |
| v_b | 0.98 | 0.99 | 0.99 | 1.00 |
| v_h | 0.94 | 0.96 | 0.98 | 0.99 |

Slope of recovered on true (1 = unbiased):

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.97 | 0.96 | 0.97 | 1.03 |
| k | 0.91 | 0.95 | 0.96 | 0.99 |
| t0 | 0.96 | 0.99 | 1.00 | 1.03 |
| v_b | 1.04 | 0.96 | 0.98 | 0.96 |
| v_h | 0.90 | 0.96 | 0.94 | 0.97 |

Largest error correlation per N (cross-talk check, |r| > 0.3 flagged):

| n_trials | param_a | param_b | error_corr |
|---|---|---|---|
| 200 | a | t0 | -0.450 |
| 500 | a | t0 | -0.519 |
| 1000 | a | t0 | -0.416 |
| 2000 | a | t0 | -0.515 |

### M3 ({200: 100, 500: 100, 1000: 100, 2000: 100} parameter vectors per N)

Pearson r:

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.96 | 0.98 | 0.99 | 0.99 |
| b_max | 0.78* | 0.82 | 0.87 | 0.85 |
| k | 0.88 | 0.94 | 0.97 | 0.98 |
| t0 | 0.97 | 0.99 | 0.99 | 0.99 |
| tau_b | 0.14* | 0.12* | 0.23* | 0.32* |
| v_h | 0.92 | 0.96 | 0.98 | 0.99 |

Slope of recovered on true (1 = unbiased):

| param \ N | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| a | 0.94 | 1.01 | 0.98 | 1.01 |
| b_max | 0.86 | 0.79 | 0.93 | 0.88 |
| k | 0.90 | 1.00 | 0.96 | 0.99 |
| t0 | 0.93 | 0.98 | 1.01 | 1.01 |
| tau_b | 0.20 | 0.14 | 0.33 | 0.45 |
| v_h | 0.91 | 1.02 | 0.95 | 0.99 |

Largest error correlation per N (cross-talk check, |r| > 0.3 flagged):

| n_trials | param_a | param_b | error_corr |
|---|---|---|---|
| 200 | b_max | tau_b | 0.487 |
| 500 | b_max | tau_b | 0.616 |
| 1000 | b_max | tau_b | 0.731 |
| 2000 | b_max | tau_b | 0.798 |
