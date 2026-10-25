# Part E: Evaluation

## Split
By time: older 80% = train, newest 20% = test, with a 24-row gap (each answer looks 24 hours ahead).

## 1. Test results (baseline: always guess Moderate = 63.4%)

| model | train_baseline_% | test_baseline_% | accuracy_% | f1_macro | poor_recall | poor_precision | auc_poor | multiclass_auc_macro |
|---|---|---|---|---|---|---|---|---|
| Decision tree | 49.4 | 63.4 | 69.4 | 0.509 | 0.134 | 0.696 | 0.792 | 0.755 |
| Naive Bayes | 49.4 | 63.4 | 54.9 | 0.508 | 0.319 | 0.521 | 0.793 | 0.76 |
| Nearest neighbour | 49.4 | 63.4 | 66.7 | 0.448 | 0.013 | 0.188 | 0.721 | 0.718 |

## 2. Why these metrics suit this user
- **Poor recall:** the user is a parent, school or outdoor worker. A Poor day with no warning is the worst mistake. Recall is the share of real Poor hours the model caught.
- **Poor precision:** if the model warns all the time, recall looks great but nobody trusts the warnings. Precision is the share of warnings that were right.
- **Macro F1:** Poor is just 15.7% of test hours, so accuracy can look fine while Poor is ignored. Macro F1 counts Good, Moderate and Poor equally.

## 3. ROC curve and AUC (Poor vs the rest)

![ROC curve](roc_curve.png)

## 4. Confusion matrices (rows = actual, columns = predicted)

### Decision tree

| Actual | Pred Good | Pred Moderate | Pred Poor |
|---|---|---|---|
| Good | 127 | 188 | 1 |
| Moderate | 56 | 892 | 13 |
| Poor | 2 | 204 | 32 |

### Naive Bayes

| Actual | Pred Good | Pred Moderate | Pred Poor |
|---|---|---|---|
| Good | 275 | 38 | 3 |
| Moderate | 414 | 480 | 67 |
| Poor | 33 | 129 | 76 |

### Nearest neighbour

| Actual | Pred Good | Pred Moderate | Pred Poor |
|---|---|---|---|
| Good | 155 | 161 | 0 |
| Moderate | 95 | 853 | 13 |
| Poor | 0 | 235 | 3 |

## 5. Cross-validation on the train pile (5 folds, time order, mean +/- spread)

| model | accuracy_% | f1_macro |
|---|---|---|
| Decision tree | 71.5 +/- 3.7 | 0.484 +/- 0.033 |
| Naive Bayes | 62.0 +/- 8.4 | 0.472 +/- 0.037 |
| Nearest neighbour | 71.1 +/- 3.0 | 0.475 +/- 0.015 |

## 6. Cost matrix and operating threshold

|  | cost |
|---|---|
| Poor day, warned | 0 |
| Poor day, not warned | 5 |
| Not Poor, warned | 1 |
| Not Poor, not warned | 0 |

The costs are our own choice: a missed Poor day is worse than a false alarm. The threshold was chosen on a validation slice from the end of the train pile, not on the test pile. The default 0.5 was not kept.

| model | chosen_threshold | missed_at_0.5 | alarms_at_0.5 | cost_at_0.5 | missed_at_chosen | alarms_at_chosen | cost_at_chosen |
|---|---|---|---|---|---|---|---|
| Decision tree | 0.05 | 206 | 14 | 1044 | 24 | 683 | 803 |
| Naive Bayes | 0.5 | 162 | 70 | 880 | 162 | 70 | 880 |
| Nearest neighbour | 0.3 | 236 | 4 | 1184 | 215 | 57 | 1132 |
