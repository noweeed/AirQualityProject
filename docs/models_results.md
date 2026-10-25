# Part D: Models

Features: CO(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), NO2(GT), PT08.S5(O3)

Settings chosen by grid search scored by f1_macro, with time-ordered 5-fold cross-validation on the training pile only.

## Test results (baseline: always guess the biggest group = 63.4%)

| model | settings | cv_f1 | accuracy | f1_macro | poor_recall |
|---|---|---|---|---|---|
| Decision tree | {'max_depth': 5, 'min_samples_leaf': 1} | 0.484 +/- 0.033 | 69.4 | 0.509 | 0.134 |
| Naive Bayes | {'var_smoothing': 0.001} | 0.472 +/- 0.037 | 54.9 | 0.508 | 0.319 |
| Nearest neighbour | {'n_neighbors': 15, 'weights': 'distance'} | 0.475 +/- 0.015 | 66.7 | 0.448 | 0.013 |

## What -200 did to the models (same target, raw vs clean features)

| model | acc_raw | acc_clean | f1_raw | f1_clean |
|---|---|---|---|---|
| Decision tree | 70.6 | 69.4 | 0.528 | 0.509 |
| Naive Bayes | 49.7 | 54.9 | 0.456 | 0.508 |
| Nearest neighbour | 67.2 | 66.7 | 0.445 | 0.448 |
