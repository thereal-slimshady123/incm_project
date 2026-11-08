# C1 results (robust variants only): 10 mice from data/processed/replication/cohort_rt2p5.parquet

## Which prior do the choices follow? (model-free logistic regression, same parameter count)

Positive dloglik = the ideal-observer (inferred) prior explains the choices better than the true block label.

| subject_id | n_trials | loglik_prior | beta_prior | loglik_prior_inferred | beta_prior_inferred | dloglik_prior_inferred_vs_prior |
|---|---|---|---|---|---|---|
| CSHL_014 | 28898 | -10481.5 | 0.45103 | -10361.8 | 0.667291 | 119.693 |
| CSHL_015 | 29885 | -10360.7 | 0.484147 | -10092.5 | 0.827558 | 268.105 |
| CSH_ZAD_005 | 14998 | -5403.77 | 0.420554 | -5334.85 | 0.644698 | 68.9227 |
| NYU-04 | 26418 | -6590.11 | 0.491499 | -6488.05 | 0.75737 | 102.057 |
| NYU-06 | 24166 | -6605.06 | 0.318708 | -6573.9 | 0.461182 | 31.1632 |
| SWC_021 | 21808 | -8684.71 | 0.359956 | -8581.87 | 0.564903 | 102.84 |
| SWC_023 | 26322 | -9481.07 | 0.47158 | -9287.24 | 0.770639 | 193.821 |
| ZM_1745 | 21415 | -5602.49 | 0.366417 | -5542.53 | 0.587259 | 59.9562 |
| ibl_witten_12 | 42735 | -10184.2 | 0.550072 | -9917.28 | 0.915386 | 266.943 |
| ibl_witten_14 | 31241 | -8849.71 | 0.62372 | -8525.67 | 1.06437 | 324.038 |

## Robust C1 (robust): wide bounds + lapse/contaminant mixture + inferred (ideal-observer) prior

30 (mouse, model) fits x up to 3 seeds; BIC re-scored on a common seed (CRN across models). Pooled run-to-run SD 110.1; decisive if > 308.1. BIC is comparable across models WITHIN this section only.

| subject_id | bic_M1 | bic_M2 | bic_M3 | dbic_M1_vs_M2 | dbic_M3_vs_M2 | winner | margin | evidence |
|---|---|---|---|---|---|---|---|---|
| CSHL_014 | 125542 | 125146 | 125426 | 396.171 | 280.474 | M2 | 280.474 | inconclusive |
| CSHL_015 | 133142 | 132505 | 132694 | 637.255 | 189.564 | M2 | 189.564 | inconclusive |
| CSH_ZAD_005 | 69460.7 | 69236.4 | 69341.9 | 224.356 | 105.515 | M2 | 105.515 | inconclusive |
| NYU-04 | 111520 | 111342 | 111399 | 177.856 | 57.0383 | M2 | 57.0383 | inconclusive |
| NYU-06 | 100802 | 100724 | 100718 | 77.9887 | -6.45879 | M3 | 6.45879 | inconclusive |
| SWC_021 | 97574.1 | 97496.9 | 97556.4 | 77.1641 | 59.4817 | M2 | 59.4817 | inconclusive |
| SWC_023 | 119231 | 118330 | 118149 | 901.601 | -180.583 | M3 | 180.583 | inconclusive |
| ZM_1745 | 89440.6 | 89309.3 | 89355 | 131.326 | 45.6995 | M2 | 45.6995 | inconclusive |
| ibl_witten_12 | 177321 | 176235 | 176485 | 1086.43 | 250.395 | M2 | 250.395 | inconclusive |
| ibl_witten_14 | 132146 | 131503 | 132007 | 643.099 | 504.246 | M2 | 504.246 | decisive |

| model | n_best | n_decisive |
|---|---|---|
| M1 | 0 | 0 |
| M2 | 8 | 1 |
| M3 | 2 | 0 |
| none (inconclusive) | nan | 9 |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 vs M2 | 10 | 435 | 310 | 235 | 655 | 0 | 10 | 0.00195 | 0.00195 | 0.00427 | 1.2 | 0.00586 | 0.00586 | 0.0128 |
| M3 vs M2 | 10 | 131 | 82.5 | 26 | 243 | 2 | 8 | 0.109 | 0.0371 | 0.0544 | 0.699 | 0.109 | 0.0371 | 0.0556 |
| M1 vs M3 | 10 | 305 | 120 | 110 | 541 | 0 | 10 | 0.00195 | 0.00195 | 0.0278 | 0.829 | 0.00586 | 0.00586 | 0.0556 |

| subset | pair | n_in_subset | mean_delta | ci_lo | ci_hi | wilcoxon_p | wilcoxon_p_holm |
|---|---|---|---|---|---|---|---|
| all mice | M1 vs M2 | 10 | 435 | 235 | 655 | 0.00195 | 0.00586 |
| all mice | M3 vs M2 | 10 | 131 | 26 | 243 | 0.0371 | 0.0371 |
| all mice | M1 vs M3 | 10 | 305 | 110 | 541 | 0.00195 | 0.00586 |

CBF agreement (RMSE):

| subject_id | M1 | M2 | M3 |
|---|---|---|---|
| CSHL_014 | 0.112 | 0.075 | 0.126 |
| CSHL_015 | 0.142 | 0.077 | 0.134 |
| CSH_ZAD_005 | 0.107 | 0.083 | 0.128 |
| NYU-04 | 0.093 | 0.078 | 0.099 |
| NYU-06 | 0.081 | 0.059 | 0.063 |
| SWC_021 | 0.096 | 0.065 | 0.096 |
| SWC_023 | 0.122 | 0.061 | 0.071 |
| ZM_1745 | 0.090 | 0.037 | 0.062 |
| ibl_witten_12 | 0.168 | 0.070 | 0.119 |
| ibl_witten_14 | 0.152 | 0.135 | 0.191 |

| pair | metric | mean_delta | n_first_closer | n_mice | wilcoxon_p | ttest_p | wilcoxon_p_holm |
|---|---|---|---|---|---|---|---|
| M1 vs M2 | rmse | 0.0425 | 0 | 10 | 0.00195 | 0.000687 | 0.00586 |
| M3 vs M2 | rmse | 0.035 | 0 | 10 | 0.00195 | 0.000331 | 0.00586 |
| M1 vs M3 | rmse | 0.0075 | 4 | 10 | 0.492 | 0.442 | 0.492 |

Calibrated CBF-slope test (parametric bootstrap; loglr > 0 favours M2; fitted parameters treated as known):

| subject_id | model | slope_obs | slope_null_mean | slope_null_sd | p_two_sided | z |
|---|---|---|---|---|---|---|
| CSHL_014 | M2 | -0.0439 | -0.0164 | 0.00571 | 0.005 | -4.8 |
| CSHL_014 | M3 | -0.0439 | 0.0221 | 0.00569 | 0.005 | -11.6 |
| CSHL_015 | M2 | -0.0526 | -0.0132 | 0.00516 | 0.005 | -7.63 |
| CSHL_015 | M3 | -0.0526 | 0.0369 | 0.00492 | 0.005 | -18.2 |
| CSH_ZAD_005 | M2 | -0.0757 | -0.0256 | 0.00825 | 0.005 | -6.07 |
| CSH_ZAD_005 | M3 | -0.0757 | 0.00571 | 0.00753 | 0.005 | -10.8 |
| NYU-04 | M2 | -0.015 | 0.0296 | 0.00575 | 0.005 | -7.75 |
| NYU-04 | M3 | -0.015 | 0.0469 | 0.00541 | 0.005 | -11.4 |
| NYU-06 | M2 | 0.000409 | 0.0136 | 0.0058 | 0.01 | -2.28 |
| NYU-06 | M3 | 0.000409 | 0.034 | 0.00578 | 0.005 | -5.81 |
| SWC_021 | M2 | -0.027 | 0.00129 | 0.00649 | 0.005 | -4.36 |
| SWC_021 | M3 | -0.027 | 0.0244 | 0.00701 | 0.005 | -7.33 |
| SWC_023 | M2 | 0.00348 | -0.00785 | 0.00587 | 0.09 | 1.93 |
| SWC_023 | M3 | 0.00348 | 0.0389 | 0.006 | 0.005 | -5.9 |
| ZM_1745 | M2 | -0.00141 | 0.0134 | 0.00608 | 0.02 | -2.44 |
| ZM_1745 | M3 | -0.00141 | 0.0392 | 0.00637 | 0.005 | -6.38 |
| ibl_witten_12 | M2 | -0.00762 | 0.00805 | 0.00433 | 0.005 | -3.62 |
| ibl_witten_12 | M3 | -0.00762 | 0.0409 | 0.00413 | 0.005 | -11.8 |
| ibl_witten_14 | M2 | -0.0764 | -0.0049 | 0.00591 | 0.005 | -12.1 |
| ibl_witten_14 | M3 | -0.0764 | 0.038 | 0.00577 | 0.005 | -19.8 |

| subject_id | loglr |
|---|---|
| CSHL_014 | 55.7 |
| CSHL_015 | 136 |
| CSH_ZAD_005 | 39.9 |
| NYU-04 | 35.2 |
| NYU-06 | 14.3 |
| SWC_021 | 17.4 |
| SWC_023 | 15.6 |
| ZM_1745 | 17.4 |
| ibl_witten_12 | 62.6 |
| ibl_witten_14 | 123 |

| pair | n_mice | sum_loglr | mean_loglr | n_favour_M2 | n_favour_M3 | wilcoxon_p |
|---|---|---|---|---|---|---|
| M2 vs M3 | 10 | 518 | 51.8 | 10 | 0 | 0.00195 |

Mixture (lapse / fast-guess) parameters:

| subject_id | model | p_lapse | p_lapse_side | p_lapse_prior |
|---|---|---|---|---|
| CSHL_014 | M1 | 0.0749 | 0.0946 | 1.59 |
| CSHL_014 | M2 | 0.0536 | -0.25 | 1.54 |
| CSHL_014 | M3 | 0.0584 | -0.212 | 1.53 |
| CSHL_015 | M1 | 0.013 | -0.882 | -0.81 |
| CSHL_015 | M2 | 0.0142 | -0.365 | -0.448 |
| CSHL_015 | M3 | 0.0138 | -1.94 | -3.02 |
| CSH_ZAD_005 | M1 | 0.225 | -1.96 | 1.44 |
| CSH_ZAD_005 | M2 | 0.22 | -2.34 | 1.7 |
| CSH_ZAD_005 | M3 | 0.211 | -2.05 | 1.45 |
| NYU-04 | M1 | 0.0913 | -0.949 | 2.46 |
| NYU-04 | M2 | 0.0948 | -0.735 | 2.37 |
| NYU-04 | M3 | 0.0821 | -0.822 | 2.06 |
| NYU-06 | M1 | 0.0882 | -0.945 | 1.6 |
| NYU-06 | M2 | 0.0914 | -0.995 | 1.69 |
| NYU-06 | M3 | 0.1 | -0.978 | 1.55 |
| SWC_021 | M1 | 0.0958 | 2.03 | 2.74 |
| SWC_021 | M2 | 0.0779 | 1.38 | 1.73 |
| SWC_021 | M3 | 0.0838 | 1.38 | 1.82 |
| SWC_023 | M1 | 0.0457 | -0.485 | 2.32 |
| SWC_023 | M2 | 0.0551 | -1.17 | 2.04 |
| SWC_023 | M3 | 0.0458 | -1.09 | 1.44 |
| ZM_1745 | M1 | 0.081 | -1.45 | 1.67 |
| ZM_1745 | M2 | 0.0907 | -1.05 | 1.41 |
| ZM_1745 | M3 | 0.0862 | -1.07 | 1.24 |
| ibl_witten_12 | M1 | 0.0324 | 1.53 | 3.04 |
| ibl_witten_12 | M2 | 0.0453 | 1.27 | 2.34 |
| ibl_witten_12 | M3 | 0.0306 | 1.17 | 2.36 |
| ibl_witten_14 | M1 | 0.0502 | -1.01 | 1.29 |
| ibl_witten_14 | M2 | 0.0621 | -1.17 | 2.15 |
| ibl_witten_14 | M3 | 0.0569 | -1.21 | 1.82 |
