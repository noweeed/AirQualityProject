# Part C: Pre-processing results

## Train / test split (by time, 24-row gap)

| part | from | to | rows |
|---|---|---|---|
| Train | 2004-03-10 | 2005-01-27 | 6153 |
| Test | 2005-01-28 | 2005-04-03 | 1515 |


## Baseline (always guess the biggest group)

| part | baseline_% |
|---|---|
| Train | 49.4 |
| Test | 63.4 |


## Rescaling decision per model

| model | rescale | reason |
|---|---|---|
| Decision Tree | No | Only asks 'is this above X?', units do not matter |
| Naive Bayes | No | Handles each column on its own |
| Nearest Neighbour | Yes | Uses distance, big columns would drown small ones |


## PCA vs feature selection (decision tree, 5-fold time CV on train, mean and spread)

| method | accuracy | accuracy_spread | f1_macro | f1_spread |
|---|---|---|---|---|
| All columns | 0.673 | 0.045 | 0.451 | 0.022 |
| Feature Selection | 0.691 | 0.05 | 0.457 | 0.024 |
| PCA | 0.671 | 0.049 | 0.439 | 0.028 |


## Best 6 columns (feature selection)

- CO(GT)
- C6H6(GT)
- PT08.S2(NMHC)
- NOx(GT)
- NO2(GT)
- PT08.S5(O3)


## Binning

Binning is not needed for these models. Decision tree, Gaussian naive Bayes and nearest neighbour all take numeric columns directly. Only the target was binned (tomorrow's NO2 into Good / Moderate / Poor).
