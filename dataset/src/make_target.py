import pandas as pd

# 1. Load the cleaned data
df = pd.read_csv("dataset/clean/airquality_clean.csv", index_col=0, parse_dates=True)

# 2. Ensure target labels are NOT derived from future-imputed values
# Only genuine 'original' NO2(GT) readings are permitted for building the target
if "NO2(GT)_src" in df.columns:
    assert (df["NO2(GT)_src"] != "interpolated").all(), "Error: NO2(GT) contains interpolated values! Targets must be genuine observations."
    raw_no2 = df["NO2(GT)"].where(df["NO2(GT)_src"] == "original")
else:
    raw_no2 = df["NO2(GT)"]

# 3. Tomorrow's NO2 = the genuine NO2 value 24 rows (24 hours) later
tomorrow = raw_no2.shift(-24)

# 4. Turn tomorrow's NO2 into a label
#    Good: up to 100 | Moderate: 101 to 200 | Poor: above 200
df["target"] = pd.cut(tomorrow, bins=[-1, 100, 200, float("inf")],
                      labels=["Good", "Moderate", "Poor"])

# 5. Remove the _src note columns (not used for the model)
df = df[[c for c in df.columns if not c.endswith("_src")]]

# 6. Drop rows where tomorrow's NO2 is missing (we never fill the target)
df = df.dropna(subset=["target"])

# 6. Show and save
print(df["target"].value_counts())
print("\nBaseline (always guess the biggest group):",
      round(df["target"].value_counts(normalize=True).max() * 100, 1), "%")
print("Rows kept:", len(df))

df.to_csv("dataset/clean/airquality_model.csv")
print("\nSaved dataset/clean/airquality_model.csv")