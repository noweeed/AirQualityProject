import os
import pandas as pd
from md_table import md_table

# Make the output folders if they don't exist
os.makedirs("docs", exist_ok=True)
os.makedirs("dataset/clean", exist_ok=True)

# 1. Load raw data (the file on disk is never changed)
df = pd.read_excel("dataset/raw/AirQualityUCI.xlsx").dropna(how="all")

# 2. Create one timestamp and sort by time
df["timestamp"] = pd.to_datetime(
    df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str)
)
df = df.drop(columns=["Date", "Time", "NMHC(GT)"])
df = df.set_index("timestamp").sort_index()

# 3. Keep a copy with -200 still in it for the before/after table
raw = df.copy()

# 4. Replace -200 with missing values
df = df.replace(-200, float("nan"))

# 5. Fill gaps of up to 3 hours and track changes
table = []

for c in list(df.columns):
    was_missing = df[c].isna()

    df[c] = df[c].interpolate(
        method="time",
        limit=3,
        limit_area="inside"
    )

    # Track whether each value was original, filled, or left empty
    src = pd.Series("original", index=df.index)
    src[was_missing & df[c].notna()] = "interpolated"
    src[was_missing & df[c].isna()] = "unrepaired"

    df[c + "_src"] = src

    table.append({
        "column": c,
        "mean_with_-200": round(raw[c].mean(), 2),
        "mean_after_repair": round(df[c].mean(), 2),
        "filled": int((src == "interpolated").sum()),
        "still_missing": int((src == "unrepaired").sum())
    })

# 6. Display and save the report
report = pd.DataFrame(table)
print(report.to_string(index=False))

with open("docs/before_after.md", "w", encoding="utf-8") as fh:
    fh.write("# Before and after repair\n\n")
    fh.write(md_table(table, list(report.columns)))
df.to_csv("dataset/clean/airquality_clean.csv")

print("\nSaved docs/before_after.md and dataset/clean/airquality_clean.csv")