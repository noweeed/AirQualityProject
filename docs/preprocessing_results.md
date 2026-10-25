# Part C: Pre-processing results

## Train / test split (by time, 24-row gap)

| part | from | to | rows |
|---|---|---|---|
| Train | 2004-03-10 | 2005-01-26 | 6464 |
| Test | 2005-01-27 | 2005-04-03 | 1592 |


## Baseline (always guess the biggest group)

| part | baseline_% |
|---|---|
| Train | 50.1 |
| Test | 63.1 |


## Rescaling decision per model

| model | rescale | reason |
|---|---|---|
| Decision Tree | No | Only asks 'is this above X?', units do not matter |
| Naive Bayes | No | Handles each column on its own |
| Nearest Neighbour | Yes | Uses distance, big columns would drown small ones |


## PCA vs feature selection (decision tree, 5-fold time CV on train, mean and spread)

| method | accuracy | accuracy_spread | f1_macro | f1_spread |
|---|---|---|---|---|
| All columns | 0.69 | 0.039 | 0.482 | 0.031 |
| Feature Selection | 0.7 | 0.057 | 0.471 | 0.035 |
| PCA | 0.672 | 0.048 | 0.436 | 0.024 |


## Best 6 columns (feature selection)

- CO(GT)
- C6H6(GT)
- PT08.S2(NMHC)
- NOx(GT)
- NO2(GT)
- PT08.S5(O3)


## Binning

Binning is not needed for these models. Decision tree, Gaussian naive Bayes and nearest neighbour all take numeric columns directly. Only the target was binned (tomorrow's NO2 into Good / Moderate / Poor).
