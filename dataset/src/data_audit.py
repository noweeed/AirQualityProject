import os
import pandas as pd
from md_table import md_table

# Load the dataset and remove completely empty rows
df = pd.read_excel("dataset/raw/AirQualityUCI.xlsx").dropna(how="all")

# Combine date and time into one column and sort the records
df["ts"] = pd.to_datetime(
    df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
    errors="coerce"
)
df = df.sort_values("ts").reset_index(drop=True)

# Treat -200 as a missing reading during checks
clean = df.replace(-200, float("nan"))

faults = []

# Add each problem and the number of affected records to the report
def add(fault, checked, count):
    faults.append({
        "fault": fault,
        "checked": checked,
        "records": int(count)
    })

# Sensor columns
sensors = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
]

# Gas measurement columns
gases = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# 1. Find rows where at least one column contains -200
add(
    "Missing readings",
    "Rows containing -200",
    (df == -200).any(axis=1).sum()
)

# 2. Count missing readings in the NMHC column
add(
    "Missing NMHC readings",
    "NMHC(GT) contains -200",
    (df["NMHC(GT)"] == -200).sum()
)

# 3. Find rows where every sensor has a missing reading
add(
    "All sensors missing",
    "All sensor readings are -200",
    (df[sensors] == -200).all(axis=1).sum()
)

# 4. Find extreme outliers
numeric = clean.select_dtypes(include="number")
q1 = numeric.quantile(0.25)
q3 = numeric.quantile(0.75)
iqr = q3 - q1

outliers = (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)

add(
    "Unusual readings",
    "Values much higher or lower than normal",
    outliers.any(axis=1).sum()
)

# 5. Find sudden changes in sensor readings between consecutive rows
jump = pd.DataFrame(False, index=clean.index, columns=sensors)

for c in sensors:
    change = clean[c].diff().abs()
    jump[c] = change > 5 * change.std()

add(
    "Sudden changes",
    "Sensor readings change sharply between hours",
    jump.any(axis=1).sum()
)

# 6. Find duplicate rows
add(
    "Duplicate rows",
    "Rows with exactly the same data",
    df.duplicated().sum()
)

# 7. Find invalid dates or times
add(
    "Invalid date or time",
    "Date and time could not be read",
    df["ts"].isna().sum()
)

# 8. Find gaps where consecutive records are not one hour apart
add(
    "Missing hours",
    "Time difference between rows is not one hour",
    (df["ts"].diff().dropna() != pd.Timedelta(hours=1)).sum()
)

# Display the report and save it as a CSV file
report = pd.DataFrame(faults)
print(report.to_string(index=False))

os.makedirs("docs", exist_ok=True)
md_rows = [{"fault": f["fault"], "detected_by": f["checked"], "records": f["records"]}
           for f in faults]
with open("docs/audit_report.md", "w", encoding="utf-8") as fh:
    fh.write("# Data quality audit report\n\n")
    fh.write(md_table(md_rows, ["fault", "detected_by", "records"]))

print("\nSaved docs/audit_report.md")