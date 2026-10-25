# Part D: Models

Features: CO(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), NO2(GT), PT08.S5(O3)

Settings chosen by grid search scored by f1_macro, with time-ordered 5-fold cross-validation on the training pile only.

## Test results (baseline: always guess the biggest group = 63.1%)

| model | settings | cv_f1 | accuracy | f1_macro | poor_recall |
|---|---|---|---|---|---|
| Decision tree | {'max_depth': 7, 'min_samples_leaf': 1} | 0.493 +/- 0.035 | 68.1 | 0.497 | 0.096 |
| Naive Bayes | {'var_smoothing': 0.001} | 0.479 +/- 0.035 | 54.8 | 0.507 | 0.317 |
| Nearest neighbour | {'n_neighbors': 41} | 0.478 +/- 0.016 | 67.8 | 0.456 | 0.0 |

## What -200 did to the models (same target, raw vs clean features)

| model | acc_raw | acc_clean | f1_raw | f1_clean |
|---|---|---|---|---|
| Decision tree | 68.8 | 68.1 | 0.478 | 0.497 |
| Naive Bayes | 52.4 | 54.8 | 0.473 | 0.507 |
| Nearest neighbour | 67.9 | 67.8 | 0.459 | 0.456 |
