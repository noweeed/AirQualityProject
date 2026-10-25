# Data quality audit report

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
