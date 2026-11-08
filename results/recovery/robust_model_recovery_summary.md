# Model recovery for the robust C1 model class (fitted parameters, each mouse's own design)

30 synthetic data sets (10 mice x generators), each fitted with M1, M2, M3 (robust settings, one fit seed, common-seed re-scoring).

Confusion matrix (rows = generating model, columns = lowest BIC):

| true | M1 | M2 | M3 |
|---|---|---|---|
| M1 | 9 | 1 | 0 |
| M2 | 1 | 7 | 2 |
| M3 | 0 | 3 | 7 |

P(recovered | true): M1 90%, M2 70%, M3 70%

Delta BIC under each generating model (positive d_M1_M2 = M1 worse than M2):

fit_model  d_M1_M2                       d_M3_M2                     
              mean median    min     max    mean median    min    max
true_model                                                           
M1          -146.4  -67.8 -402.3    23.4    84.0   85.3  -32.3  200.9
M2           416.8  227.0   -8.8  1151.4   148.9   86.1   -6.5  421.2
M3           350.4  199.1  -18.1   967.6   -80.7  -43.2 -310.2   21.2

Observed (real mice, robust variant): mean dBIC M1-M2 = 320.0, M3-M2 = 5.0
