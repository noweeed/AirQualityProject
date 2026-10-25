import os
import pandas as pd
from md_table import md_table
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate

os.makedirs("docs", exist_ok=True)

# 1. Load the cleaned data
df = pd.read_csv("dataset/clean/airquality_model.csv", index_col=0, parse_dates=True)

# 2. Split data by time: 80% for training, 20% for testing
# Leave a 24-row gap because the target is 24 hours ahead
cut = int(len(df) * 0.8)
train = df.iloc[:cut]
test = df.iloc[cut + 24:]

train.to_csv("dataset/clean/train.csv")
test.to_csv("dataset/clean/test.csv")

# Separate input columns and target
X_train = train.drop(columns="target")
y_train = train["target"]
X_test = test.drop(columns="target")
y_test = test["target"]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training dates:", train.index.min().date(), "to", train.index.max().date())
print("Testing dates:", test.index.min().date(), "to", test.index.max().date())

# Check the baseline accuracy
print("\n--- BASELINE ---")
print("Train:", round(y_train.value_counts(normalize=True).max() * 100, 1), "%")
print("Test:", round(y_test.value_counts(normalize=True).max() * 100, 1), "%")

# 3. Decide whether scaling is needed
print("\n--- SCALING ---")
print("Decision Tree: Not needed")
print("Naive Bayes: Not needed")
print("Nearest Neighbour: Needed")

# 4. Compare PCA and feature selection using a decision tree
# Fill missing values using the median
K = 6
fill = SimpleImputer(strategy="median")
tree = DecisionTreeClassifier(max_depth=5, random_state=42)

options = {
    "All columns": make_pipeline(fill, tree),
    "Feature Selection": make_pipeline(
        fill, SelectKBest(f_classif, k=K), tree
    ),
    "PCA": make_pipeline(
        fill, StandardScaler(), PCA(n_components=K), tree
    ),
}

# Test each method using the training data
cv = TimeSeriesSplit(n_splits=5)

print("\n--- PCA VS FEATURE SELECTION ---")

rows = []

for name, pipe in options.items():
    results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"]
    )

    accuracy = results["test_accuracy"]
    f1 = results["test_f1_macro"]

    print(f"\n{name}")
    print("Accuracy:", round(accuracy.mean(), 3))
    print("F1 Score:", round(f1.mean(), 3))

    rows.append({
        "method": name,
        "accuracy": round(accuracy.mean(), 3),
        "accuracy_spread": round(accuracy.std(), 3),
        "f1_macro": round(f1.mean(), 3),
        "f1_spread": round(f1.std(), 3)
    })

# 5. Find the best 6 columns
filled = SimpleImputer(strategy="median").fit_transform(X_train)

picker = SelectKBest(f_classif, k=K)
picker.fit(filled, y_train)

chosen = list(X_train.columns[picker.get_support()])

print("\n--- SELECTED FEATURES ---")
print(chosen)

# 6. Decide whether binning is needed
print("\n--- BINNING ---")
print("Binning is not needed for these models.")

# 7. Save everything for Part C in one markdown file
train_base = round(y_train.value_counts(normalize=True).max() * 100, 1)
test_base = round(y_test.value_counts(normalize=True).max() * 100, 1)

md = ["# Part C: Pre-processing results\n"]

md.append("## Train / test split (by time, 24-row gap)\n")
md.append(md_table([
    {"part": "Train", "from": train.index.min().date(), "to": train.index.max().date(), "rows": len(train)},
    {"part": "Test", "from": test.index.min().date(), "to": test.index.max().date(), "rows": len(test)},
], ["part", "from", "to", "rows"]))

md.append("\n## Baseline (always guess the biggest group)\n")
md.append(md_table([
    {"part": "Train", "baseline_%": train_base},
    {"part": "Test", "baseline_%": test_base},
], ["part", "baseline_%"]))

md.append("\n## Rescaling decision per model\n")
md.append(md_table([
    {"model": "Decision Tree", "rescale": "No", "reason": "Only asks 'is this above X?', units do not matter"},
    {"model": "Naive Bayes", "rescale": "No", "reason": "Handles each column on its own"},
    {"model": "Nearest Neighbour", "rescale": "Yes", "reason": "Uses distance, big columns would drown small ones"},
], ["model", "rescale", "reason"]))

md.append(f"\n## PCA vs feature selection (decision tree, {cv.n_splits}-fold time CV on train, mean and spread)\n")
md.append(md_table(rows, ["method", "accuracy", "accuracy_spread", "f1_macro", "f1_spread"]))

md.append(f"\n## Best {K} columns (feature selection)\n")
md.append("\n".join(f"- {c}" for c in chosen) + "\n")

md.append("\n## Binning\n")
md.append("Binning is not needed for these models. Decision tree, Gaussian naive Bayes and "
          "nearest neighbour all take numeric columns directly. Only the target was binned "
          "(tomorrow's NO2 into Good / Moderate / Poor).\n")

with open("docs/preprocessing_results.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))
with open("docs/part_c_results.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))

print("\nAll files saved successfully.")