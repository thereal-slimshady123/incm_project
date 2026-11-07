# C2 bound-type recovery (M2; truth known)

| config | truth | n_datasets | n_weibull_wins | n_weibull_wins_by_10 | mean_dbic | median_lam | median_frac_t1 | median_frac_t2.5 |
|---|---|---|---|---|---|---|---|---|
| 2.5s_c1_likelihood | fixed | 4 | 1 | 1 | 13.3 | 2.73 | 0.902 | 0.438 |
| 2.5s_c1_likelihood | fixed+slow_lapses | 4 | 1 | 1 | -18.5 | 11.2 | 0.951 | 0.728 |
| 2.5s_c1_likelihood | weibull | 4 | 4 | 4 | -110 | 2.51 | 0.889 | 0.372 |
| 2.5s_truncated | fixed | 4 | 2 | 2 | 4.32 | 5.35 | 0.945 | 0.697 |
| 2.5s_truncated | fixed+slow_lapses | 4 | 3 | 2 | -17.8 | 6.44 | 0.941 | 0.809 |
| 2.5s_truncated | weibull | 4 | 4 | 4 | -76.6 | 3.41 | 0.818 | 0.464 |
| 5.0s_truncated | fixed | 4 | 1 | 0 | 9.36 | 14.3 | 1 | 0.995 |
| 5.0s_truncated | fixed+slow_lapses | 4 | 0 | 0 | 18.2 | 19.8 | 1 | 1 |
| 5.0s_truncated | weibull | 4 | 4 | 4 | -104 | 3.63 | 0.856 | 0.506 |
