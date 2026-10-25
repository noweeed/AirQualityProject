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
    
    # 12 strict assertions verifying data hygiene and physical validity
    assert not (df[vals] == -200).any().any(), "Check 1 failed: -200 sentinels found in dataset"
    assert "NMHC(GT)" not in df.columns, "Check 2 failed: NMHC(GT) column was not dropped"
    assert df["CO(GT)"].min() >= 0, f"Check 3 failed: CO(GT) contains negative readings (min: {df['CO(GT)'].min()})"
    assert df["C6H6(GT)"].min() >= 0, f"Check 4 failed: C6H6(GT) contains negative readings (min: {df['C6H6(GT)'].min()})"
    assert df["NOx(GT)"].min() >= 0, f"Check 5 failed: NOx(GT) contains negative readings (min: {df['NOx(GT)'].min()})"
    assert df["NO2(GT)"].min() >= 0, f"Check 6 failed: NO2(GT) contains negative readings (min: {df['NO2(GT)'].min()})"
    assert df[sens].min().min() >= 0, f"Check 7 failed: cheap sensor readings contain negative values (min: {df[sens].min().min()})"
    assert df["RH"].min() >= 0 and df["RH"].max() <= 100, f"Check 8 failed: humidity RH out of range [0, 100]"
    assert df["T"].min() >= -30 and df["T"].max() <= 50, f"Check 9 failed: temperature T out of range [-30, 50]"
    assert df["AH"].min() >= 0, f"Check 10 failed: absolute humidity AH is negative"
    assert not df.index.duplicated().any(), "Check 11 failed: duplicate timestamps found"
    assert df.index.is_monotonic_increasing, "Check 12 failed: timestamps are not in chronological order"
    return True

if __name__ == "__main__":
    print("=== VALIDATION SUITE ===")

    print("\n1. Testing RAW data (verifying assertions catch corruptions):")
    raw_failed_as_expected = False
    try:
        validate(load_raw())
        print("FAIL: Raw data unexpectedly passed validation!")
    except AssertionError as err:
        raw_failed_as_expected = True
        print(f"PASS (as expected): Raw data raised AssertionError:\n  {err}")
    assert raw_failed_as_expected, "Validation suite must reject raw data!"

    print("\n2. Testing CLEAN data (verifying all 12 assertions pass):")
    validate(load_clean())
    print("PASS: Clean data passed all 12 strict assertions with zero errors!")