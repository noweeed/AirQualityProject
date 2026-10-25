import pandas as pd

# Load raw data (with -200) and cleaned data
def load_raw():
    df = pd.read_excel("dataset/raw/AirQualityUCI.xlsx").dropna(how="all")
    df.index = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str))
    return df.drop(columns=["Date", "Time"]).sort_index()

def load_clean():
    return pd.read_csv("dataset/clean/airquality_clean.csv", index_col=0, parse_dates=True)

sens = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]

def validate(df):
    # every column that is a real reading (not a _src column)
    vals = [c for c in df.columns if not c.endswith("_src")]
    # min() skips missing values, so NaN never causes a false fail
    checks = {
        "1. no -200 left in any reading":      not (df[vals] == -200).any().any(),
        "2. NMHC(GT) column dropped":          "NMHC(GT)" not in df.columns,
        "3. CO(GT) never below 0":             df["CO(GT)"].min() >= 0,
        "4. C6H6(GT) never below 0":           df["C6H6(GT)"].min() >= 0,
        "5. NOx(GT) never below 0":            df["NOx(GT)"].min() >= 0,
        "6. NO2(GT) never below 0":            df["NO2(GT)"].min() >= 0,
        "7. cheap sensors never below 0":      df[sens].min().min() >= 0,
        "8. humidity RH between 0 and 100":    df["RH"].min() >= 0 and df["RH"].max() <= 100,
        "9. temperature T between -30 and 50": df["T"].min() >= -30 and df["T"].max() <= 50,
        "10. absolute humidity AH never below 0": df["AH"].min() >= 0,
        "11. no duplicate timestamps":         not df.index.duplicated().any(),
        "12. rows are in time order":          df.index.is_monotonic_increasing,
    }
    return checks

for name, data in [("RAW", load_raw()), ("CLEAN", load_clean())]:
    results = validate(data)
    passed = sum(results.values())
    print(f"\n=== {name} data: {passed} of {len(results)} passed ===")
    for check, ok in results.items():
        print("  PASS" if ok else "  FAIL", check)