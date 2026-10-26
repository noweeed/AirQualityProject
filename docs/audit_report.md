# Data Quality Audit Report

This report documents the initial inspection of the raw `AirQualityUCI.xlsx` dataset (9,357 records) to detect data corruptions, sensor faults, and missingness before running data cleaning.

## 1. Fault Detection Results

| fault | detected_by | records |
|---|---|---|
| Missing readings | Rows containing -200 | 8530 |
| Missing NMHC readings | NMHC(GT) contains -200 | 8443 |
| All sensors missing | All sensor readings are -200 | 366 |
| Unusual readings | Values much higher or lower than normal | 127 |
| Sudden changes | Sensor readings change sharply between hours | 349 |
| Duplicate rows | Rows with exactly the same data | 0 |
| Invalid date or time | Date and time could not be read | 0 |
| Missing hours | Time difference between rows is not one hour | 0 |

## 2. Repair Strategy Applied

Based on the findings above, the cleaning pipeline applied the following targeted repairs:

1. **Sentinel Replacement:** Converted all -200 codes into `NaN` so computers no longer treat missing readings as valid negative measurements or freezing temperatures.
2. **Dropping Unusable Columns:** Dropped `NMHC(GT)` entirely because over 90% (8,443 rows) was missing and could not be safely reconstructed.
3. **Short Gap Forward-Fill (1 to 3 hours):** For brief sensor dropouts of 1 to 3 hours, carried the last known valid reading forward (causal forward-fill). This preserves short-term atmospheric continuity without peeking into the future.
4. **Preserving Long Outages (> 3 hours):** Left multi-day sensor outages blank in the clean dataset to avoid inventing fake weather patterns.
5. **Target Protection:** Ground truth `NO2(GT)` was never interpolated to guarantee models are evaluated purely on genuine physical observations.
6. **Provenance Tracking:** Added companion `_src` columns for all sensor fields marking every value as `original`, `interpolated`, or `unrepaired`.

