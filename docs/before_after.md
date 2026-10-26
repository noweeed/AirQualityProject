# Before and After Data Repair
 
This table demonstrates the quantitative effect of repairing -200 sentinels and applying causal forward-fill for brief outages.
 
## Numeric Comparison
 
| column | mean_with_-200 | mean_after_repair | filled | still_missing |
|---|---|---|---|---|
| CO(GT) | -34.21 | 2.13 | 166 | 1517 |
| PT08.S1(CO) | 1048.87 | 1099.79 | 6 | 360 |
| C6H6(GT) | 1.87 | 10.08 | 6 | 360 |
| PT08.S2(NMHC) | 894.48 | 939.12 | 6 | 360 |
| NOx(GT) | 168.6 | 242.07 | 325 | 1314 |
| PT08.S3(NOx) | 794.87 | 835.31 | 6 | 360 |
| NO2(GT) | 58.14 | 113.08 | 0 | 1642 |
| PT08.S4(NO2) | 1391.36 | 1456.25 | 6 | 360 |
| PT08.S5(O3) | 974.95 | 1022.85 | 6 | 360 |
| T | 9.78 | 18.32 | 6 | 360 |
| RH | 39.48 | 49.23 | 6 | 360 |
| AH | -6.84 | 1.03 | 6 | 360 |
 
## Column Details and Key Observations
 
- **`mean_with_-200`:** Shows how -200 sentinels heavily dragged down physical averages. For instance, `CO(GT)` had an impossible raw mean of -34.21 mg/m³ and `AH` (absolute humidity) was -6.84.
- **`mean_after_repair`:** True physical mean after converting -200 to NaN and forward-filling short gaps (`CO(GT)` corrected to +2.13 mg/m³, `AH` corrected to +1.03).
- **`filled`:** Total hourly gaps of 1 to 3 hours repaired using causal forward-fill (e.g. 166 readings in `CO(GT)`, 325 in `NOx(GT)`). Note that `NO2(GT)` has 0 filled because target source values are never interpolated.
- **`still_missing`:** Outages longer than 3 hours left alone during cleaning to prevent inventing fake weather. These are handled as neutral medians during model training.
- **Provenance Tracking:** In `airquality_clean.csv`, every sensor column has a companion `*_src` column recording whether each cell is `original`, `interpolated`, or `unrepaired`.

