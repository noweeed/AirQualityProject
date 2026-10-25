import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score

os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. The 6 columns that feature selection picked in Part C
FEATURES = ["CO(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)", "PT08.S5(O3)"]

# 2. Load the same train and test piles for every model
train = pd.read_csv("dataset/clean/train.csv", index_col=0, parse_dates=True)
test = pd.read_csv("dataset/clean/test.csv", index_col=0, parse_dates=True)
X_train, y_train = train[FEATURES], train["target"]
X_test, y_test = test[FEATURES], test["target"]
baseline = y_test.value_counts(normalize=True).max() * 100

# 3. The 3 models. Each one fills missing values with the TRAIN median first.
#    Only nearest neighbour is rescaled (it uses distance).
fill = ("fill", SimpleImputer(strategy="median"))
models = {
    "Decision tree": (
        Pipeline([fill, ("model", DecisionTreeClassifier(random_state=42))]),
        {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [1, 20, 50]}),
    "Naive Bayes": (
        Pipeline([fill, ("model", GaussianNB())]),
        {"model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
    "Nearest neighbour": (
        Pipeline([fill, ("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
        {"model__n_neighbors": [3, 5, 7, 11, 15, 21], "model__weights": ["distance", "uniform"]}),
}

# 4. Choose the settings by grid search, scored by f1_macro,
#    with cross-validation on the TRAIN pile only, in time order
cv = TimeSeriesSplit(n_splits=5)
best = {}
rows = []
for name, (pipe, grid) in models.items():
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro")
    search.fit(X_train, y_train)
    best[name] = search.best_estimator_
    i = search.best_index_
    cv_mean = search.cv_results_["mean_test_score"][i]
    cv_std = search.cv_results_["std_test_score"][i]

    # 5. Score on the test pile (first time the test pile is used)
    pred = search.best_estimator_.predict(X_test)
    rows.append({
        "model": name,
        "settings": str(search.best_params_).replace("model__", ""),
        "cv_f1": f"{cv_mean:.3f} +/- {cv_std:.3f}",
        "accuracy": round(accuracy_score(y_test, pred) * 100, 1),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 3),
        "poor_recall": round(recall_score(y_test, pred, labels=["Poor"], average=None)[0], 3),
    })
    joblib.dump(search.best_estimator_, f"models/{name.replace(' ', '_').lower()}.joblib")

results = pd.DataFrame(rows)
print(f"Test baseline (always guess the biggest group): {baseline:.1f}%\n")
print(results.to_string(index=False))

# 6. YOUR ASSIGNED CHALLENGE: what did -200 do to the models?
#    Same models, same settings, same target. Only the FEATURES change:
#    raw readings (with -200 left in) vs cleaned readings.
raw = pd.read_excel("dataset/raw/AirQualityUCI.xlsx").dropna(how="all")
raw.index = pd.to_datetime(raw["Date"].dt.strftime("%Y-%m-%d") + " " + raw["Time"].astype(str))
raw = raw.sort_index()
Xr_train, Xr_test = raw.loc[train.index, FEATURES], raw.loc[test.index, FEATURES]

print("\n-200 CHALLENGE: same models, raw features (-200 left in) vs clean features")
challenge = []
for name, model in best.items():
    model.fit(Xr_train, y_train)                 # learns from the raw numbers
    pred_raw = model.predict(Xr_test)
    model.fit(X_train, y_train)                  # put it back on clean data
    pred_clean = model.predict(X_test)
    challenge.append({
        "model": name,
        "acc_raw": round(accuracy_score(y_test, pred_raw) * 100, 1),
        "acc_clean": round(accuracy_score(y_test, pred_clean) * 100, 1),
        "f1_raw": round(f1_score(y_test, pred_raw, average="macro"), 3),
        "f1_clean": round(f1_score(y_test, pred_clean, average="macro"), 3),
    })
challenge = pd.DataFrame(challenge)
print(challenge.to_string(index=False))

# 7. Save everything in one markdown file
def table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)
    return head + body + "\n"

with open("docs/models_results.md", "w", encoding="utf-8") as f:
    f.write("# Part D: Models\n\n")
    f.write(f"Features: {', '.join(FEATURES)}\n\n")
    f.write("Settings chosen by grid search scored by f1_macro, with time-ordered 5-fold ")
    f.write("cross-validation on the training pile only.\n\n")
    f.write(f"## Test results (baseline: always guess the biggest group = {baseline:.1f}%)\n\n")
    f.write(table(results))
    f.write("\n## What -200 did to the models (same target, raw vs clean features)\n\n")
    f.write(table(challenge))

print("\nSaved docs/models_results.md and models/*.joblib")