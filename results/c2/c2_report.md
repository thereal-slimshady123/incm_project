# C2 results: fixed vs Weibull-collapsing bounds, and invariance of the C1 conclusion

## Variant `trunc_2p5`: cutoff 2.5 s, truncation-aware likelihood

20 (mouse, model, bound) fits x up to 4 replicates (independent seeds plus polish passes); best-BIC replicate kept. Run-to-run SD of BIC between the independent seeds = 218.3; decisive if |delta BIC| > 611.3.

### M2: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 81994.8 | 81888.2 | -106.655 | -63.15 | weibull | inconclusive | 18.5209 | 2.10766 | 0.997873 | 0.985421 | False |
| CSHL046 | 55606.7 | 55512 | -94.7639 | -56.8118 | weibull | inconclusive | 3.37649 | 3.25541 | 0.981142 | 0.686662 | False |
| CSH_ZAD_002 | 66943.3 | 66962 | 18.6828 | -0.237679 | fixed | inconclusive | 20 | 4 | 0.999994 | 0.999756 | True |
| KS019 | 161986 | 162009 | 22.8921 | 0.955553 | fixed | inconclusive | 19.9945 | 3.99311 | 0.999994 | 0.999752 | True |
| KS021 | 56713.9 | 56730.8 | 16.8328 | -1.02632 | fixed | inconclusive | 2.54125 | 2.43804 | 0.902204 | 0.382554 | False |
| NYU-01 | 76975 | 76996.3 | 21.2736 | 0.867459 | fixed | inconclusive | 19.984 | 3.27343 | 0.999945 | 0.998892 | True |
| NYU-09 | 94027.9 | 94046.9 | 18.9152 | -0.554427 | fixed | inconclusive | 19.9692 | 3.46438 | 0.999969 | 0.999253 | True |
| ZM_1743 | 67783.2 | 67736 | -47.1441 | -33.2224 | weibull | inconclusive | 3.15064 | 2.96083 | 0.967108 | 0.60402 | False |
| ibl_witten_07 | 175259 | 175280 | 21.5205 | 0.156943 | fixed | inconclusive | 19.9878 | 3.98516 | 0.999993 | 0.999748 | True |
| ibl_witten_13 | 169689 | 169711 | 22.5426 | 0.684294 | fixed | inconclusive | 19.9978 | 3.9973 | 0.999994 | 0.999754 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | -10.6 | 18.8 | -44.2 | 19.5 | 3 | 7 | 0.344 | 1 | 0.536 | -0.203 | 0.344 | 1 | 0.536 | M2 |

## Variant `trunc_5p0`: cutoff 5.0 s, truncation-aware likelihood

20 (mouse, model, bound) fits x up to 4 replicates (independent seeds plus polish passes); best-BIC replicate kept. Run-to-run SD of BIC between the independent seeds = 385.1; decisive if |delta BIC| > 1078.2.

### M2: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 88561.6 | 88540.6 | -21.0174 | -20.3873 | weibull | inconclusive | 19.9865 | 3.5567 | 0.999976 | 0.999385 | True |
| CSHL046 | 66826.7 | 66840.4 | 13.7042 | -2.71115 | fixed | inconclusive | 19.9968 | 3.59386 | 0.999979 | 0.999432 | True |
| CSH_ZAD_002 | 81779.2 | 81793.9 | 14.7028 | -2.38498 | fixed | inconclusive | 19.9957 | 3.99465 | 0.999994 | 0.999753 | True |
| KS019 | 172818 | 172830 | 11.933 | -4.57532 | fixed | inconclusive | 19.9804 | 3.9764 | 0.999993 | 0.999743 | True |
| KS021 | 58278.8 | 58298.1 | 19.3121 | 0.196515 | fixed | inconclusive | 19.9892 | 3.98576 | 0.999993 | 0.999748 | True |
| NYU-01 | 89790.3 | 89795.6 | 5.31347 | -7.22451 | fixed | inconclusive | 19.5984 | 3.5338 | 0.999973 | 0.999309 | False |
| NYU-09 | 105169 | 105183 | 13.7477 | -3.22232 | fixed | inconclusive | 19.9931 | 3.99555 | 0.999994 | 0.999753 | True |
| ZM_1743 | 76281.1 | 76304.8 | 23.754 | 2.14185 | fixed | inconclusive | 19.9708 | 3.9604 | 0.999993 | 0.999733 | True |
| ibl_witten_07 | 189046 | 189067 | 21.3059 | 0.00416446 | fixed | inconclusive | 19.9993 | 3.99915 | 0.999994 | 0.999755 | True |
| ibl_witten_13 | 179962 | 179980 | 18.2952 | -1.47635 | fixed | inconclusive | 19.9799 | 3.48624 | 0.999971 | 0.999287 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | 12.1 | 14.2 | 3.47 | 18.1 | 1 | 9 | 0.0215 | 0.0488 | 0.015 | 0.949 | 0.0215 | 0.0488 | 0.015 | M2 |

### RT tail (model M2; model RTs from the mice's own trials replayed, cut to the same window)

Model-free descriptive tests on the mice (the hazard test is inflated by truncation at the cutoff; read the data-minus-model tests below instead):

| test | n_mice | mean | median | n_positive | wilcoxon_p |
|---|---|---|---|---|---|
| log hazard ratio (late/early) vs 0 [truncation-inflated, descriptive] | 10 | 0.314 | 0.261 | 10 | 0.00195 |
| accuracy(<1 s) - accuracy(>2 s) vs 0 | 10 | 0.178 | 0.183 | 10 | 0.00195 |

Data minus model (cut to the same window), tested against 0 across mice; positive = the mice have a higher late hazard ratio / late accuracy / slow-tail fraction than the model:

| bound | metric | n_mice | mean | median | n_positive | wilcoxon_p | ttest_p |
|---|---|---|---|---|---|---|---|
| fixed | log_hr_diff | 10 | 0.408 | 0.355 | 10 | 0.00195 | 1.18e-05 |
| fixed | accuracy_diff | 10 | 0.167 | 0.156 | 9 | 0.00391 | 0.000609 |
| fixed | p_gt2p5_diff | 10 | 0.0469 | 0.0345 | 10 | 0.00195 | 0.000177 |
| weibull | log_hr_diff | 10 | 0.409 | 0.354 | 10 | 0.00195 | 1.03e-05 |
| weibull | accuracy_diff | 10 | 0.166 | 0.156 | 9 | 0.00391 | 0.000581 |
| weibull | p_gt2p5_diff | 10 | 0.047 | 0.0346 | 10 | 0.00195 | 0.000174 |

Distance between model and mouse tail, Weibull minus fixed (negative = collapsing closer):

| metric | n_mice | mean_fixed | mean_weibull | mean_delta | n_weibull_closer | wilcoxon_p | ttest_p |
|---|---|---|---|---|---|---|---|
| survival_error | 10 | 0.0352 | 0.0353 | 0.000143 | 5 | 0.625 | 0.258 |
| hazard_log_err | 10 | 0.408 | 0.409 | 0.000973 | 5 | 0.922 | 0.498 |
| accuracy_error | 10 | 0.168 | 0.167 | -0.00102 | 4 | 0.695 | 0.519 |

| subject_id | bound | survival_error | hazard_ratio_data | hazard_ratio_model | p_gt2p5_data | p_gt2p5_model | accuracy_gt2s_data | accuracy_gt2s_model |
|---|---|---|---|---|---|---|---|---|
| CSHL045 | fixed | 0.0264 | 1.23 | 0.939 | 0.0546 | 0.0184 | 0.753 | 0.659 |
| CSHL045 | weibull | 0.0268 | 1.23 | 0.939 | 0.0546 | 0.0182 | 0.753 | 0.66 |
| CSHL046 | fixed | 0.0613 | 1.33 | 0.964 | 0.125 | 0.05 | 0.669 | 0.604 |
| CSHL046 | weibull | 0.0612 | 1.33 | 0.963 | 0.125 | 0.0501 | 0.669 | 0.604 |
| CSH_ZAD_002 | fixed | 0.068 | 1.62 | 0.978 | 0.146 | 0.0591 | 0.883 | 0.6 |
| CSH_ZAD_002 | weibull | 0.0681 | 1.62 | 0.978 | 0.146 | 0.0591 | 0.883 | 0.6 |
| KS019 | fixed | 0.0192 | 1.27 | 0.954 | 0.05 | 0.0202 | 0.863 | 0.579 |
| KS019 | weibull | 0.0193 | 1.27 | 0.942 | 0.05 | 0.0201 | 0.863 | 0.577 |
| KS021 | fixed | 0.0143 | 1.25 | 0.615 | 0.0167 | 6.24e-05 | 0.796 | 0.531 |
| KS021 | weibull | 0.0143 | 1.25 | 0.616 | 0.0167 | 6.24e-05 | 0.796 | 0.545 |
| NYU-01 | fixed | 0.0561 | 1.64 | 0.959 | 0.106 | 0.0291 | 0.763 | 0.582 |
| NYU-01 | weibull | 0.056 | 1.64 | 0.96 | 0.106 | 0.0292 | 0.763 | 0.581 |
| NYU-09 | fixed | 0.0224 | 1.26 | 0.922 | 0.0807 | 0.0479 | 0.779 | 0.648 |
| NYU-09 | weibull | 0.0224 | 1.26 | 0.923 | 0.0807 | 0.0479 | 0.779 | 0.648 |
| ZM_1743 | fixed | 0.041 | 1.23 | 0.953 | 0.0812 | 0.0284 | 0.75 | 0.635 |
| ZM_1743 | weibull | 0.0421 | 1.23 | 0.951 | 0.0812 | 0.0274 | 0.75 | 0.633 |
| ibl_witten_07 | fixed | 0.0213 | 1.6 | 0.966 | 0.0445 | 0.0127 | 0.562 | 0.568 |
| ibl_witten_07 | weibull | 0.0213 | 1.6 | 0.967 | 0.0445 | 0.0127 | 0.562 | 0.568 |
| ibl_witten_13 | fixed | 0.0218 | 1.36 | 0.922 | 0.0363 | 0.00583 | 0.841 | 0.581 |
| ibl_witten_13 | weibull | 0.0218 | 1.36 | 0.925 | 0.0363 | 0.00585 | 0.841 | 0.582 |

## Variant `c1like_2p5`: cutoff 2.5 s, C1 (untruncated) likelihood

60 (mouse, model, bound) fits x up to 4 replicates (independent seeds plus polish passes); best-BIC replicate kept. Run-to-run SD of BIC between the independent seeds = 258.4; decisive if |delta BIC| > 723.6.

### M1: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 81796.5 | 81810.4 | 13.9479 | -2.84847 | fixed | inconclusive | 19.9566 | 3.9454 | 0.999993 | 0.999724 | True |
| CSHL046 | 55250.4 | 55267.1 | 16.7791 | -1.04033 | fixed | inconclusive | 19.9748 | 3.97495 | 0.999993 | 0.999742 | True |
| CSH_ZAD_002 | 66522.5 | 66539.4 | 16.9261 | -1.11605 | fixed | inconclusive | 19.981 | 3.93633 | 0.999992 | 0.99972 | True |
| KS019 | 162526 | 162547 | 21.1538 | 0.0864152 | fixed | inconclusive | 19.9787 | 3.97977 | 0.999993 | 0.999744 | True |
| KS021 | 57224.3 | 57244.1 | 19.8394 | 0.476962 | fixed | inconclusive | 19.9886 | 3.98573 | 0.999993 | 0.999748 | True |
| NYU-01 | 76681.4 | 76699.7 | 18.2989 | -0.619856 | fixed | inconclusive | 19.9823 | 3.97782 | 0.999993 | 0.999743 | True |
| NYU-09 | 93761.3 | 93781.9 | 20.5324 | 0.254155 | fixed | inconclusive | 19.9845 | 3.9806 | 0.999993 | 0.999745 | True |
| ZM_1743 | 67357.4 | 67383.1 | 25.6961 | 3.19763 | fixed | inconclusive | 19.8372 | 3.41679 | 0.999963 | 0.999156 | True |
| ibl_witten_07 | 176135 | 176142 | 6.39963 | -7.4035 | fixed | inconclusive | 19.9727 | 3.97097 | 0.999993 | 0.999739 | True |
| ibl_witten_13 | 170567 | 170587 | 20.5819 | -0.296076 | fixed | inconclusive | 19.9979 | 3.9974 | 0.999994 | 0.999754 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | 18 | 19.1 | 14.7 | 20.8 | 0 | 10 | 0.00195 | 0.00195 | 1.55e-06 | 3.49 | 0.00195 | 0.00195 | 1.55e-06 | M1 |

### M2: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 81700.8 | 81717.7 | 16.8257 | -1.4096 | fixed | inconclusive | 19.9931 | 3.9892 | 0.999994 | 0.99975 | True |
| CSHL046 | 55211.9 | 55159 | -52.967 | -35.9134 | weibull | inconclusive | 10.1299 | 3.54316 | 0.999727 | 0.992995 | False |
| CSH_ZAD_002 | 66301 | 66318.4 | 17.438 | -0.860088 | fixed | inconclusive | 19.9967 | 3.99593 | 0.999994 | 0.999754 | True |
| KS019 | 161992 | 162014 | 22.1055 | 0.562249 | fixed | inconclusive | 19.9794 | 3.47804 | 0.99997 | 0.999275 | True |
| KS021 | 56746.4 | 56766.8 | 20.3613 | 0.737906 | fixed | inconclusive | 19.9804 | 3.98085 | 0.999993 | 0.999745 | True |
| NYU-01 | 76830.1 | 76856.5 | 26.3328 | 3.39705 | fixed | inconclusive | 19.9761 | 3.96644 | 0.999993 | 0.999737 | True |
| NYU-09 | 93638 | 93657.8 | 19.8508 | -0.0866218 | fixed | inconclusive | 19.999 | 3.99877 | 0.999994 | 0.999755 | True |
| ZM_1743 | 67309.1 | 67328.6 | 19.4487 | 0.0739402 | fixed | inconclusive | 20 | 4 | 0.999994 | 0.999756 | True |
| ibl_witten_07 | 175323 | 175357 | 34.1585 | 6.47595 | fixed | inconclusive | 19.9831 | 3.97043 | 0.999993 | 0.99974 | True |
| ibl_witten_13 | 169587 | 169607 | 20.1075 | -0.53324 | fixed | inconclusive | 19.9985 | 3.99811 | 0.999994 | 0.999755 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | 14.4 | 20 | -1.86 | 24.2 | 1 | 9 | 0.0215 | 0.084 | 0.0932 | 0.594 | 0.0215 | 0.084 | 0.0932 | M2 |

### M3: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 81711.7 | 81737.5 | 25.7672 | 3.06117 | fixed | inconclusive | 19.9776 | 3.0164 | 0.999881 | 0.998108 | True |
| CSHL046 | 55190.9 | 55210 | 19.0958 | 0.118019 | fixed | inconclusive | 19.9865 | 3.01572 | 0.999881 | 0.998108 | True |
| CSH_ZAD_002 | 66273.7 | 66272.5 | -1.1625 | -10.1603 | weibull | inconclusive | 19.841 | 3.62166 | 0.99998 | 0.999448 | True |
| KS019 | 161486 | 161508 | 21.8909 | 0.454936 | fixed | inconclusive | 19.9645 | 3.96954 | 0.999993 | 0.999738 | True |
| KS021 | 56788.8 | 56808 | 19.22 | 0.167286 | fixed | inconclusive | 19.985 | 3.98123 | 0.999993 | 0.999745 | True |
| NYU-01 | 76626.4 | 76645.3 | 18.8796 | -0.329511 | fixed | inconclusive | 19.9836 | 3.98271 | 0.999993 | 0.999746 | True |
| NYU-09 | 93754.8 | 93723.1 | -31.6251 | -25.8246 | weibull | inconclusive | 19.9626 | 3.89521 | 0.999991 | 0.999694 | True |
| ZM_1743 | 67320.8 | 67355.3 | 34.4825 | 7.59086 | fixed | inconclusive | 19.7416 | 3.72002 | 0.999985 | 0.999541 | True |
| ibl_witten_07 | 175483 | 175504 | 21.5706 | 0.182009 | fixed | inconclusive | 19.9732 | 3.50708 | 0.999973 | 0.999317 | True |
| ibl_witten_13 | 169785 | 169790 | 5.29637 | -7.93883 | fixed | inconclusive | 19.9528 | 3.94782 | 0.999993 | 0.999725 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | 13.3 | 19.2 | 1.03 | 22.8 | 2 | 8 | 0.109 | 0.084 | 0.0502 | 0.715 | 0.109 | 0.084 | 0.0502 | M3 |

### Invariance of the C1 comparison (M1 / M2 / M3) under the Weibull bound

| subject_id | dbic_M1_vs_M2_fixed | dbic_M1_vs_M2_weibull | dbic_M3_vs_M2_fixed | dbic_M3_vs_M2_weibull | winner_fixed | winner_weibull | same_winner |
|---|---|---|---|---|---|---|---|
| CSHL045 | 95.6306 | 92.7529 | 10.8639 | 19.8054 | M2 | M2 | True |
| CSHL046 | 38.4409 | 108.187 | -21.062 | 51.0008 | M3 | M2 | False |
| CSH_ZAD_002 | 221.47 | 220.958 | -27.2915 | -45.8919 | M3 | M3 | True |
| KS019 | 533.607 | 532.655 | -505.637 | -505.851 | M3 | M3 | True |
| KS021 | 477.868 | 477.346 | 42.3633 | 41.222 | M2 | M2 | True |
| NYU-01 | -148.693 | -156.727 | -203.698 | -211.151 | M3 | M3 | True |
| NYU-09 | 123.339 | 124.021 | 116.781 | 65.3053 | M2 | M2 | True |
| ZM_1743 | 48.2778 | 54.5251 | 11.7112 | 26.745 | M2 | M2 | True |
| ibl_witten_07 | 812.435 | 784.677 | 159.738 | 147.15 | M2 | M2 | True |
| ibl_witten_13 | 979.665 | 980.14 | 197.512 | 182.701 | M2 | M2 | True |

Across-mouse tests under each bound, and tests of the shift between bounds (bound = shift):

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | bound | rel_mean | rel_ci_lo | rel_ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 vs M2 | 10 | 318 | 172 | 117 | 542 | 1 | 9 | 0.0215 | 0.0195 | 0.0231 | 0.864 | 0.043 | 0.0391 | 0.0518 | fixed | nan | nan | nan |
| M3 vs M2 | 10 | -21.9 | 11.3 | -150 | 82 | 4 | 6 | 0.754 | 0.922 | 0.742 | -0.107 | 0.754 | 0.922 | 0.742 | fixed | nan | nan | nan |
| M1 vs M3 | 10 | 340 | 167 | 137 | 569 | 0 | 10 | 0.00195 | 0.00195 | 0.0173 | 0.921 | 0.00586 | 0.00586 | 0.0518 | fixed | nan | nan | nan |
| M1 vs M2 | 10 | 322 | 172 | 124 | 540 | 1 | 9 | 0.0215 | 0.0195 | 0.0197 | 0.895 | 0.043 | 0.0391 | 0.0468 | weibull | nan | nan | nan |
| M3 vs M2 | 10 | -22.9 | 34 | -150 | 77.9 | 3 | 7 | 0.344 | 0.695 | 0.726 | -0.114 | 0.344 | 0.695 | 0.726 | weibull | nan | nan | nan |
| M1 vs M3 | 10 | 345 | 170 | 145 | 573 | 0 | 10 | 0.00195 | 0.00195 | 0.0156 | 0.94 | 0.00586 | 0.00586 | 0.0468 | weibull | nan | nan | nan |
| M1 vs M2 | 10 | 3.65 | -0.517 | -8.2 | 20.6 | 6 | 4 | nan | 0.625 | 0.655 | nan | nan | nan | nan | shift | 0.0115 | -0.0258 | 0.0647 |
| M3 vs M2 | 10 | -1.02 | -4.3 | -18 | 18.9 | 7 | 3 | nan | 0.557 | 0.92 | nan | nan | nan | nan | shift | -0.0468 | -0.824 | 0.863 |
| M1 vs M3 | 10 | 4.67 | -0.659 | -5.38 | 17.5 | 6 | 4 | nan | 0.922 | 0.474 | nan | nan | nan | nan | shift | 0.0137 | -0.0158 | 0.0516 |

Rank agreement of the per-mouse deltas between bounds:

| pair | spearman_rho | p | same_sign | n_mice |
|---|---|---|---|---|
| M1 vs M2 | 0.964 | 7.32e-06 | 10 | 10 |
| M3 vs M2 | 0.927 | 0.000112 | 9 | 10 |
| M1 vs M3 | 0.927 | 0.000112 | 10 | 10 |

C1 verdict under the fixed bound: {'m1_rejected': True, 'm2_vs_m3': 'not separable', 'cbf_closest': 'M2'}

C1 verdict under the Weibull bound: {'m1_rejected': True, 'm2_vs_m3': 'not separable', 'cbf_closest': 'M2'}

**Invariant: True**

## Variant `tail_5p0`: cutoff 5.0 s, truncation-aware likelihood

20 (mouse, model, bound) fits x up to 4 replicates (independent seeds plus polish passes); best-BIC replicate kept. Run-to-run SD of BIC between the independent seeds = 452.0; decisive if |delta BIC| > 1265.7.

### M2: Weibull vs fixed bound (delta BIC = Weibull - fixed; negative = collapsing better)

| subject_id | bic_fixed | bic_weibull | dbic_weibull_vs_fixed | dnll_weibull_vs_fixed | winner | evidence | p_lam | p_gamma | frac_t1 | frac_t2.5 | lam_at_ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSHL045 | 93271.9 | 93292.9 | 21.0512 | 0.646951 | fixed | inconclusive | 20 | 4 | 0.999994 | 0.999756 | True |
| CSHL046 | 70132 | 69922.3 | -209.682 | -114.404 | weibull | inconclusive | 5.2979 | 3.75516 | 0.998093 | 0.942147 | False |
| CSH_ZAD_002 | 85692.9 | 85194.3 | -498.544 | -259.008 | weibull | inconclusive | 4.88843 | 3.27685 | 0.994498 | 0.894856 | False |
| KS019 | 182270 | 182260 | -9.97553 | -15.5296 | weibull | inconclusive | 19.6897 | 3.96622 | 0.999993 | 0.999721 | True |
| KS021 | 62463.5 | 62482.5 | 19.0577 | 0.0692953 | fixed | inconclusive | 19.9963 | 3.99538 | 0.999994 | 0.999753 | True |
| NYU-01 | 94365.9 | 94474.2 | 108.254 | 44.2456 | fixed | inconclusive | 19.8995 | 3.6579 | 0.999982 | 0.999494 | True |
| NYU-09 | 110439 | 110039 | -399.719 | -209.956 | weibull | inconclusive | 4.8613 | 3.51254 | 0.996137 | 0.907807 | False |
| ZM_1743 | 80030.1 | 80046.6 | 16.4529 | -1.50867 | fixed | inconclusive | 19.9768 | 3.96994 | 0.999993 | 0.999739 | True |
| ibl_witten_07 | 199402 | 199495 | 92.3744 | 35.5384 | fixed | inconclusive | 19.9928 | 3.35671 | 0.999957 | 0.999069 | True |
| ibl_witten_13 | 192113 | 192029 | -84.1888 | -52.7183 | weibull | inconclusive | 19.969 | 3.27806 | 0.999945 | 0.9989 | True |

| pair | n_mice | mean_delta | median_delta | ci_lo | ci_hi | n_first_better | n_second_better | sign_p | wilcoxon_p | ttest_p | dz | sign_p_holm | wilcoxon_p_holm | ttest_p_holm | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull vs fixed | 10 | -94.5 | 3.24 | -224 | 18.9 | 5 | 5 | 1 | 0.625 | 0.186 | -0.453 | 1 | 0.625 | 0.186 | M2 |

### RT tail (model M2; model RTs from the mice's own trials replayed, cut to the same window)

Model-free descriptive tests on the mice (the hazard test is inflated by truncation at the cutoff; read the data-minus-model tests below instead):

| test | n_mice | mean | median | n_positive | wilcoxon_p |
|---|---|---|---|---|---|
| log hazard ratio (late/early) vs 0 [truncation-inflated, descriptive] | 10 | 0.314 | 0.261 | 10 | 0.00195 |
| accuracy(<1 s) - accuracy(>2 s) vs 0 | 10 | 0.178 | 0.183 | 10 | 0.00195 |

Data minus model (cut to the same window), tested against 0 across mice; positive = the mice have a higher late hazard ratio / late accuracy / slow-tail fraction than the model:

| bound | metric | n_mice | mean | median | n_positive | wilcoxon_p | ttest_p |
|---|---|---|---|---|---|---|---|
| fixed | log_hr_diff | 10 | 0.332 | 0.311 | 10 | 0.00195 | 3.73e-05 |
| fixed | accuracy_diff | 10 | 0.163 | 0.158 | 9 | 0.00391 | 0.000541 |
| fixed | p_gt2p5_diff | 10 | 0.0421 | 0.0296 | 10 | 0.00195 | 0.000489 |
| weibull | log_hr_diff | 10 | 0.306 | 0.267 | 10 | 0.00195 | 6.99e-05 |
| weibull | accuracy_diff | 10 | 0.17 | 0.165 | 9 | 0.00391 | 0.000287 |
| weibull | p_gt2p5_diff | 10 | 0.0395 | 0.0272 | 10 | 0.00195 | 0.000353 |

Distance between model and mouse tail, Weibull minus fixed (negative = collapsing closer):

| metric | n_mice | mean_fixed | mean_weibull | mean_delta | n_weibull_closer | wilcoxon_p | ttest_p |
|---|---|---|---|---|---|---|---|
| survival_error | 10 | 0.0312 | 0.0295 | -0.00177 | 7 | 0.084 | 0.0736 |
| hazard_log_err | 10 | 0.332 | 0.306 | -0.026 | 7 | 0.16 | 0.093 |
| accuracy_error | 10 | 0.165 | 0.172 | 0.0069 | 5 | 0.322 | 0.147 |

| subject_id | bound | survival_error | hazard_ratio_data | hazard_ratio_model | p_gt2p5_data | p_gt2p5_model | accuracy_gt2s_data | accuracy_gt2s_model |
|---|---|---|---|---|---|---|---|---|
| CSHL045 | fixed | 0.0193 | 1.23 | 0.973 | 0.0546 | 0.0257 | 0.753 | 0.651 |
| CSHL045 | weibull | 0.0193 | 1.23 | 0.973 | 0.0546 | 0.0257 | 0.753 | 0.651 |
| CSHL046 | fixed | 0.0608 | 1.33 | 0.962 | 0.125 | 0.0496 | 0.669 | 0.615 |
| CSHL046 | weibull | 0.0542 | 1.33 | 1.03 | 0.125 | 0.0573 | 0.669 | 0.576 |
| CSH_ZAD_002 | fixed | 0.0642 | 1.62 | 0.988 | 0.146 | 0.0636 | 0.883 | 0.612 |
| CSH_ZAD_002 | weibull | 0.0578 | 1.62 | 1.08 | 0.146 | 0.0741 | 0.883 | 0.606 |
| KS019 | fixed | 0.0136 | 1.27 | 0.954 | 0.05 | 0.0309 | 0.863 | 0.592 |
| KS019 | weibull | 0.0135 | 1.27 | 0.955 | 0.05 | 0.0311 | 0.863 | 0.593 |
| KS021 | fixed | 0.0126 | 1.25 | 1.16 | 0.0167 | 0.000351 | 0.796 | 0.556 |
| KS021 | weibull | 0.0126 | 1.25 | 1.15 | 0.0167 | 0.000351 | 0.796 | 0.553 |
| NYU-01 | fixed | 0.0503 | 1.64 | 0.976 | 0.106 | 0.0347 | 0.763 | 0.583 |
| NYU-01 | weibull | 0.0494 | 1.64 | 0.961 | 0.106 | 0.0355 | 0.763 | 0.582 |
| NYU-09 | fixed | 0.0206 | 1.26 | 0.936 | 0.0807 | 0.0505 | 0.779 | 0.654 |
| NYU-09 | weibull | 0.017 | 1.26 | 1.04 | 0.0807 | 0.0568 | 0.779 | 0.63 |
| ZM_1743 | fixed | 0.0371 | 1.23 | 0.968 | 0.0812 | 0.0336 | 0.75 | 0.614 |
| ZM_1743 | weibull | 0.037 | 1.23 | 0.969 | 0.0812 | 0.0337 | 0.75 | 0.615 |
| ibl_witten_07 | fixed | 0.0165 | 1.6 | 0.971 | 0.0445 | 0.0197 | 0.562 | 0.573 |
| ibl_witten_07 | weibull | 0.0166 | 1.6 | 0.973 | 0.0445 | 0.0196 | 0.562 | 0.572 |
| ibl_witten_13 | fixed | 0.0172 | 1.36 | 0.953 | 0.0363 | 0.011 | 0.841 | 0.578 |
| ibl_witten_13 | weibull | 0.0174 | 1.36 | 0.963 | 0.0363 | 0.0107 | 0.841 | 0.58 |

## Agreement of the per-mouse delta BIC (Weibull - fixed, model M2) between variants

| variant_a | variant_b | n_mice | spearman_rho | spearman_p | n_same_sign | mean_a | mean_b |
|---|---|---|---|---|---|---|---|
| trunc_2p5 | trunc_5p0 | 10 | 0.0909 | 0.803 | 8 | -10.6 | 12.1 |
| trunc_2p5 | c1like_2p5 | 10 | 0.77 | 0.00922 | 8 | -10.6 | 14.4 |
| trunc_2p5 | tail_5p0 | 10 | 0.0303 | 0.934 | 4 | -10.6 | -94.5 |
| trunc_5p0 | c1like_2p5 | 10 | 0.2 | 0.58 | 8 | 12.1 | 14.4 |
| trunc_5p0 | tail_5p0 | 10 | -0.0788 | 0.829 | 4 | 12.1 | -94.5 |
| c1like_2p5 | tail_5p0 | 10 | 0.564 | 0.0897 | 6 | 14.4 | -94.5 |
