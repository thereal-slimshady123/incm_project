# C1 results: M1 (start point) vs M2 (drift bias) vs M3 (dynamic bias)

Cohort: 10 mice, RT cutoff 2.5 s, fixed bounds. Fits: 30 (mouse, model) x 2 replicate seeds; best-BIC replicate kept.

Run-to-run noise: pooled SD of BIC = 252.9 (median replicate spread 124.4, max 1041.4). A difference is *decisive* if > 708.1 BIC.

## Per-mouse BIC (dBIC < 0: better than M2)

| subject_id | bic_M1 | bic_M2 | bic_M3 | dbic_M1_vs_M2 | dbic_M3_vs_M2 | winner | margin | evidence |
|---|---|---|---|---|---|---|---|---|
| CSHL045 | 81796.5 | 81700.8 | 81711.7 | 95.6306 | 10.8639 | M2 | 10.8639 | inconclusive |
| CSHL046 | 55250.4 | 55211.9 | 55225.5 | 38.4409 | 13.5836 | M2 | 13.5836 | inconclusive |
| CSH_ZAD_002 | 66579.1 | 66301 | 66273.7 | 278.07 | -27.2915 | M3 | 27.2915 | inconclusive |
| KS019 | 162526 | 161992 | 161486 | 533.607 | -505.637 | M3 | 505.637 | inconclusive |
| KS021 | 57321 | 56746.4 | 56857.4 | 574.558 | 110.94 | M2 | 110.94 | inconclusive |
| NYU-01 | 76908.7 | 76830.1 | 76787.1 | 78.5264 | -42.9966 | M3 | 42.9966 | inconclusive |
| NYU-09 | 93761.3 | 93638 | 93754.8 | 123.339 | 116.781 | M2 | 116.781 | inconclusive |
| ZM_1743 | 67357.4 | 67342.5 | 67320.8 | 14.9237 | -21.6429 | M3 | 21.6429 | inconclusive |
| ibl_witten_07 | 176254 | 175323 | 175483 | 931.178 | 159.738 | M2 | 159.738 | inconclusive |
| ibl_witten_13 | 170613 | 169587 | 169785 | 1025.85 | 197.512 | M2 | 197.512 | inconclusive |

## Winners

| model | n_best | n_decisive |
|---|---|---|
| M1 | 0 | 0 |
| M2 | 6 | 0 |
| M3 | 4 | 0 |
| none (inconclusive) | nan | 10 |

## Across-mouse tests on BIC (negative mean_delta favours the first model; Holm over 3 pairs)

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 vs M2 | 10 | 369 | 201 | 165 | 597 | 0 | 10 | 0.00195 | 0.00195 | 0.0128 | 0.98 | 0.00586 | 0.00586 | 0.0383 |
| M3 vs M2 | 10 | 1.19 | 12.2 | -126 | 96.8 | 4 | 6 | 0.754 | 0.625 | 0.985 | 0.00601 | 0.754 | 0.625 | 0.985 |
| M1 vs M3 | 10 | 368 | 213 | 153 | 606 | 0 | 10 | 0.00195 | 0.00195 | 0.0145 | 0.955 | 0.00586 | 0.00586 | 0.0383 |

## Same tests on mouse subsets

| subset | pair | n_in_subset | mean_delta | ci_lo | ci_hi | wilcoxon_p | wilcoxon_p_holm | ttest_p |
|---|---|---|---|---|---|---|---|---|
| all mice | M1 vs M2 | 10 | 369 | 165 | 597 | 0.00195 | 0.00586 | 0.0128 |
| all mice | M3 vs M2 | 10 | 1.19 | -126 | 96.8 | 0.625 | 0.625 | 0.985 |
| all mice | M1 vs M3 | 10 | 368 | 153 | 606 | 0.00195 | 0.00586 | 0.0145 |
| excl. lapse-affected (KS019, KS021, CSH_ZAD_002, NYU-01) | M1 vs M2 | 6 | 372 | 55 | 702 | 0.0312 | 0.0938 | 0.112 |
| excl. lapse-affected (KS019, KS021, CSH_ZAD_002, NYU-01) | M3 vs M2 | 6 | 79.5 | 18.1 | 147 | 0.156 | 0.156 | 0.0844 |
| excl. lapse-affected (KS019, KS021, CSH_ZAD_002, NYU-01) | M1 vs M3 | 6 | 292 | 30.7 | 561 | 0.0312 | 0.0938 | 0.13 |
| excl. KS021 | M1 vs M2 | 9 | 347 | 123 | 599 | 0.00391 | 0.0117 | 0.0293 |
| excl. KS021 | M3 vs M2 | 9 | -11 | -152 | 93.4 | 0.91 | 0.91 | 0.876 |
| excl. KS021 | M1 vs M3 | 9 | 358 | 128 | 624 | 0.00391 | 0.0117 | 0.03 |

## CBF agreement (RMSE, lower = closer to the data)

| subject_id | M1 | M2 | M3 |
|---|---|---|---|
| CSHL045 | 0.192 | 0.113 | 0.150 |
| CSHL046 | 0.161 | 0.079 | 0.062 |
| CSH_ZAD_002 | 0.139 | 0.057 | 0.101 |
| KS019 | 0.187 | 0.084 | 0.097 |
| KS021 | 0.179 | 0.052 | 0.066 |
| NYU-01 | 0.210 | 0.082 | 0.108 |
| NYU-09 | 0.096 | 0.069 | 0.065 |
| ZM_1743 | 0.138 | 0.089 | 0.124 |
| ibl_witten_07 | 0.166 | 0.070 | 0.136 |
| ibl_witten_13 | 0.168 | 0.059 | 0.106 |

| pair | metric | mean_delta | n_first_closer | n_mice | wilcoxon_p | ttest_p | wilcoxon_p_holm |
|---|---|---|---|---|---|---|---|
| M1 vs M2 | rmse | 0.0881 | 0 | 10 | 0.00195 | 1.1e-05 | 0.00586 |
| M3 vs M2 | rmse | 0.026 | 2 | 10 | 0.0195 | 0.00957 | 0.0195 |
| M1 vs M3 | rmse | 0.062 | 0 | 10 | 0.00195 | 0.000403 | 0.00586 |
| M1 vs M2 | chi2_per_bin | 57.9 | 0 | 10 | 0.00195 | 0.00172 | 0.00586 |
| M3 vs M2 | chi2_per_bin | 15.7 | 1 | 10 | 0.00586 | 0.0287 | 0.00586 |
| M1 vs M3 | chi2_per_bin | 42.1 | 0 | 10 | 0.00195 | 0.00325 | 0.00586 |

## Sensitivity (no_history): refit WITHOUT the choice-history drift v_h (one seed); BIC not comparable with the main table unless the model is unchanged

| subject_id | bic_M1 | bic_M2 | bic_M3 | dbic_M1_vs_M2 | dbic_M3_vs_M2 | winner |
|---|---|---|---|---|---|---|
| CSHL045 | 82716.1 | 81763.1 | 82297.9 | 953.01 | 534.856 | M2 |
| CSHL046 | 55645.2 | 55233.7 | 55262.9 | 411.536 | 29.2321 | M2 |
| CSH_ZAD_002 | 66372.5 | 66397 | 65883.8 | -24.5235 | -513.191 | M3 |
| KS019 | 165360 | 163393 | 163084 | 1967.08 | -309.464 | M3 |
| KS021 | 58862.4 | 57562.6 | 58005.4 | 1299.76 | 442.849 | M2 |
| NYU-01 | 78444.4 | 77356.7 | 77219.4 | 1087.71 | -137.237 | M3 |
| NYU-09 | 94370.5 | 94072.7 | 93842.4 | 297.779 | -230.324 | M3 |
| ZM_1743 | 68695.3 | 67747.8 | 67945.4 | 947.482 | 197.622 | M2 |
| ibl_witten_07 | 178327 | 176055 | 176738 | 2272.46 | 683.162 | M2 |
| ibl_witten_13 | 173495 | 171268 | 172069 | 2227.46 | 800.806 | M2 |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 vs M2 | 10 | 1.14e+03 | 1.02e+03 | 684 | 1.61e+03 | 1 | 9 | 0.0215 | 0.00391 | 0.00151 | 1.42 | 0.043 | 0.00781 | 0.00303 |
| M3 vs M2 | 10 | 150 | 113 | -109 | 411 | 4 | 6 | 0.754 | 0.375 | 0.322 | 0.332 | 0.754 | 0.375 | 0.322 |
| M1 vs M3 | 10 | 994 | 803 | 657 | 1.39e+03 | 0 | 10 | 0.00195 | 0.00195 | 0.000696 | 1.59 | 0.00586 | 0.00586 | 0.00209 |

| subject_id | M1 | M2 | M3 |
|---|---|---|---|
| CSHL045 | 0.278 | 0.136 | 0.161 |
| CSHL046 | 0.230 | 0.099 | 0.073 |
| CSH_ZAD_002 | 0.188 | 0.030 | 0.121 |
| KS019 | 0.282 | 0.141 | 0.140 |
| KS021 | 0.330 | 0.080 | 0.195 |
| NYU-01 | 0.310 | 0.130 | 0.117 |
| NYU-09 | 0.166 | 0.096 | 0.064 |
| ZM_1743 | 0.199 | 0.124 | 0.116 |
| ibl_witten_07 | 0.223 | 0.039 | 0.131 |
| ibl_witten_13 | 0.274 | 0.077 | 0.158 |

## Sensitivity (wide_bounds): refit with RELAXED search bounds t0 in [0.02, 0.6], s_z in [0, 1.2]; BIC not comparable with the main table unless the model is unchanged

| subject_id | bic_M1 | bic_M2 | bic_M3 | dbic_M1_vs_M2 | dbic_M3_vs_M2 | winner |
|---|---|---|---|---|---|---|
| CSHL045 | 81326.3 | 80932.3 | 81126.5 | 394.023 | 194.205 | M2 |
| CSHL046 | 54902.8 | 54788.9 | 54807.7 | 113.941 | 18.8459 | M2 |
| CSH_ZAD_002 | 66397.6 | 66287.5 | 66256.2 | 110.138 | -31.2604 | M3 |
| KS019 | 163168 | 162083 | 161375 | 1084.94 | -707.949 | M3 |
| KS021 | 57206.6 | 56776.2 | 56800.1 | 430.415 | 23.9845 | M2 |
| NYU-01 | 76732.1 | 76417.9 | 76469.2 | 314.187 | 51.2749 | M2 |
| NYU-09 | 93625.3 | 93642.6 | 93562.6 | -17.353 | -80.0832 | M3 |
| ZM_1743 | 67084.9 | 67044.5 | 67085.7 | 40.4771 | 41.2035 | M2 |
| ibl_witten_07 | 175701 | 174262 | 174541 | 1438.64 | 279.033 | M2 |
| ibl_witten_13 | 170653 | 169439 | 169597 | 1213.66 | 157.954 | M2 |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 vs M2 | 10 | 512 | 354 | 227 | 836 | 1 | 9 | 0.0215 | 0.00391 | 0.0141 | 0.96 | 0.0645 | 0.0117 | 0.0423 |
| M3 vs M2 | 10 | -5.28 | 32.6 | -181 | 124 | 3 | 7 | 0.344 | 0.432 | 0.952 | -0.0196 | 0.344 | 0.432 | 0.952 |
| M1 vs M3 | 10 | 518 | 231 | 195 | 907 | 1 | 9 | 0.0215 | 0.00391 | 0.0243 | 0.855 | 0.0645 | 0.0117 | 0.0486 |

| subject_id | M1 | M2 | M3 |
|---|---|---|---|
| CSHL045 | 0.249 | 0.116 | 0.156 |
| CSHL046 | 0.161 | 0.105 | 0.064 |
| CSH_ZAD_002 | 0.124 | 0.068 | 0.096 |
| KS019 | 0.171 | 0.039 | 0.090 |
| KS021 | 0.194 | 0.042 | 0.063 |
| NYU-01 | 0.185 | 0.088 | 0.101 |
| NYU-09 | 0.116 | 0.082 | 0.072 |
| ZM_1743 | 0.134 | 0.108 | 0.109 |
| ibl_witten_07 | 0.178 | 0.070 | 0.140 |
| ibl_witten_13 | 0.186 | 0.061 | 0.079 |

## Follow-up: combined start point + drift bias (JOINT, 1 extra parameter) vs M1/M2/M3

| subject_id | bic_M2 | bic_JOINT | dbic_JOINT_vs_M2 | dbic_JOINT_vs_M3 | p_z | p_v_b | winner |
|---|---|---|---|---|---|---|---|
| CSHL045 | 81700.8 | 82361.6 | 660.75 | 649.886 | -0.0243901 | 0.621458 | M2 |
| CSHL046 | 55211.9 | 55398.7 | 186.817 | 173.233 | 0.0155655 | 0.247588 | M2 |
| CSH_ZAD_002 | 66301 | 66447 | 145.982 | 173.274 | -0.0335683 | 0.433507 | M3 |
| KS019 | 161992 | 161420 | -571.848 | -66.2108 | -0.139898 | 1.22161 | JOINT |
| KS021 | 56746.4 | 56871.2 | 124.75 | 13.8098 | -0.00864415 | 0.698358 | M2 |
| NYU-01 | 76830.1 | 76900.3 | 70.1423 | 113.139 | -0.0727883 | 0.597292 | M3 |
| NYU-09 | 93638 | 93970 | 332.025 | 215.244 | -0.0359893 | 0.369752 | M2 |
| ZM_1743 | 67342.5 | 67371.8 | 29.283 | 50.9259 | -0.0277007 | 0.144236 | M3 |
| ibl_witten_07 | 175323 | 175713 | 389.835 | 230.097 | -0.0485769 | 0.488298 | M2 |
| ibl_witten_13 | 169587 | 169820 | 232.886 | 35.3737 | -0.116726 | 1.34352 | M2 |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JOINT vs M2 | 10 | 160 | 166 | -36.4 | 332 | 1 | 9 | 0.0215 | 0.0645 | 0.144 | 0.506 | 0.0645 | 0.129 | 0.288 |
| JOINT vs M1 | 10 | -209 | -70.2 | -511 | 82.1 | 6 | 4 | 0.754 | 0.432 | 0.224 | -0.413 | 0.754 | 0.432 | 0.288 |
| JOINT vs M3 | 10 | 159 | 143 | 59.2 | 284 | 1 | 9 | 0.0215 | 0.0137 | 0.0314 | 0.805 | 0.0645 | 0.041 | 0.0942 |

CBF discrepancy (RMSE), JOINT vs the others:

| pair | metric | mean_delta | n_first_closer | n_mice | wilcoxon_p | ttest_p | wilcoxon_p_holm |
|---|---|---|---|---|---|---|---|
| JOINT vs M2 | rmse | 0.0087 | 4 | 10 | 0.16 | 0.107 | 0.16 |
| JOINT vs M1 | rmse | -0.0794 | 10 | 10 | 0.00195 | 7.74e-05 | 0.00586 |
| JOINT vs M3 | rmse | -0.0173 | 8 | 10 | 0.0645 | 0.063 | 0.129 |

## BIC winner vs empirical CBF shape

| subject_id | winner | evidence | emp_shape | emp_slope | winner_shape | shape_agrees | cbf_closest |
|---|---|---|---|---|---|---|---|
| CSHL045 | M2 | inconclusive | falling | -0.04 | flat | False | M2 |
| CSHL046 | M2 | inconclusive | flat | 0.00782 | flat | True | M3 |
| CSH_ZAD_002 | M3 | inconclusive | falling | -0.0332 | rising | False | M2 |
| KS019 | M3 | inconclusive | falling | -0.0376 | rising | False | M2 |
| KS021 | M2 | inconclusive | flat | -0.0232 | flat | True | M2 |
| NYU-01 | M3 | inconclusive | flat | -0.0237 | rising | False | M2 |
| NYU-09 | M2 | inconclusive | flat | 0.0123 | flat | True | M3 |
| ZM_1743 | M3 | inconclusive | flat | -0.0286 | rising | False | M2 |
| ibl_witten_07 | M2 | inconclusive | falling | -0.0412 | flat | False | M2 |
| ibl_witten_13 | M2 | inconclusive | flat | -0.0141 | flat | True | M2 |
