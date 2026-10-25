# Part A: Data and Documentation

## 1. Dataset
- Source: UCI Air Quality dataset
- Records: 9,357 
- Attributes: 15 
- Measurement types present: nominal, ordinal, interval and ratio

## 2. Data dictionary

| Column | What it is | Type | Plausible range | Why this range | Role |
|---|---|---|---|---|---|
| Date | Calendar date | Interval | Mar 2004 to Apr 2005 | Dates of the study | Identifier |
| Time | Hour of the day | Interval | 0 to 23 | 24 hours in a day | Identifier |
| CO(GT) | Carbon monoxide, lab analyser | Ratio | 0 to 15 | Can't be negative; real max is 11.9 | Explanatory |
| PT08.S1(CO) | Cheap CO sensor | Ratio | 0 to 2500 | Sensor signal; real max is 2040 | Explanatory |
| NMHC(GT) | Hydrocarbons, lab analyser | Ratio | 0 to 1500 | Can't be negative; real max is 1189 | **Must not be used** (about 90% missing) |
| C6H6(GT) | Benzene | Ratio | 0 to 100 | Can't be negative; real max is 63.7 | Explanatory |
| PT08.S2(NMHC) | Cheap hydrocarbon sensor | Ratio | 0 to 2500 | Sensor signal; real max is 2214 | Explanatory |
| NOx(GT) | Nitrogen oxides, lab analyser | Ratio | 0 to 1500 | Can't be negative; real max is 1479 | Explanatory |
| PT08.S3(NOx) | Cheap NOx sensor | Ratio | 0 to 3000 | Sensor signal; real max is 2683 | Explanatory |
| NO2(GT) | Nitrogen dioxide, lab analyser | Ratio | 0 to 400 | Can't be negative; real max is 340 | Explanatory (current hour) and source of the target (24 hours later) |
| PT08.S4(NO2) | Cheap NO2 sensor | Ratio | 0 to 3000 | Sensor signal; real max is 2775 | Explanatory |
| PT08.S5(O3) | Cheap ozone sensor | Ratio | 0 to 3000 | Sensor signal; real max is 2523 | Explanatory |
| T | Temperature in °C | Interval | -10 to 50 | Italian weather | Explanatory |
| RH | Relative humidity in % | Ratio | 0 to 100 | It is a percentage | Explanatory |
| AH | Absolute humidity | Ratio | 0 to 5 | Real max is 2.2 | Explanatory |
| hour (created) | Hour from Time | Ordinal | 0 to 23 | Same as Time | Explanatory |
| day_of_week (created) | Monday to Sunday | Nominal | 7 values | Calendar | Explanatory |
| season (created) | Winter, spring, summer, autumn | Nominal | 4 values | Calendar | Explanatory |
| target (created) | Air quality category 24 hours ahead | Ordinal | Good < Moderate < Poor | Made from NO2 | **Target** |
| *_src (created in cleaning) | How each value was repaired | Nominal | original / interpolated / unrepaired | Cleaning log | **Must not be used** as a feature |

Note: -200 is not a real value in any column. It means "no reading" and is replaced by missing before any range is checked.

## 3. Columns not known when the user asks

The user asks at hour t. Only readings from hour t and earlier are known at that moment. So these are excluded:

1. **The target itself:** the NO2 category at hour t+24. It is what we are predicting.
2. **Any reading after hour t:** for example, tomorrow's temperature or sensor values.
3. **Anything built from future hours:** for example, a daily average that includes hours after t. Only averages that look backwards are allowed.
4. **The *_src columns:** they describe how we cleaned the data, which the user would not have.

NMHC(GT) is also excluded, for a different reason: it is about 90% missing and cannot be repaired.