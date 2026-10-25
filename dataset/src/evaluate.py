import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw pictures without opening a window
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, confusion_matrix)

os.makedirs("docs", exist_ok=True)

FEATURES = ["CO(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)", "PT08.S5(O3)"]
CLASSES = ["Good", "Moderate", "Poor"]

# small helper: turn a table into markdown text
def md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values)
    return head + body + "\n"

# 1. Load the two piles and the 3 trained models
train = pd.read_csv("dataset/clean/train.csv", index_col=0, parse_dates=True)
test = pd.read_csv("dataset/clean/test.csv", index_col=0, parse_dates=True)
X_train, y_train = train[FEATURES], train["target"]
X_test, y_test = test[FEATURES], test["target"]

models = {
    "Decision tree": joblib.load("models/decision_tree.joblib"),
    "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    "Nearest neighbour": joblib.load("models/nearest_neighbour.joblib"),
}

# 2. Baseline = always guess the biggest group (computed from the real data)
base_train = y_train.value_counts(normalize=True)
base_test = y_test.value_counts(normalize=True)
baseline = round(base_test.max() * 100, 1)
print(f"Test baseline: always guess {base_test.idxmax()} = {baseline}%")
print(f"Share of Poor hours: train {base_train.get('Poor', 0)*100:.1f}%, test {base_test.get('Poor', 0)*100:.1f}%")

# 3. Test results: accuracy + extra metrics, baselines, and multiclass AUC
rows, probs = [], {}
is_poor = (y_test == "Poor").values
for name, m in models.items():
    pred = m.predict(X_test)
    poor_idx = list(m.classes_).index("Poor")
    probs[name] = m.predict_proba(X_test)[:, poor_idx]
    all_probs = m.predict_proba(X_test)
    rows.append({
        "model": name,
        "train_baseline_%": round(base_train.max() * 100, 1),
        "test_baseline_%": baseline,
        "accuracy_%": round(accuracy_score(y_test, pred) * 100, 1),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 3),
        "poor_recall": round(recall_score(y_test, pred, labels=["Poor"], average=None, zero_division=0)[0], 3),
        "poor_precision": round(precision_score(y_test, pred, labels=["Poor"], average=None, zero_division=0)[0], 3),
        "auc_poor": round(roc_auc_score(is_poor, probs[name]), 3),
        "multiclass_auc_macro": round(roc_auc_score(y_test, all_probs, multi_class="ovr", average="macro", labels=CLASSES), 3),
    })
perf = pd.DataFrame(rows)
print("\n=== TEST RESULTS ===")
print(perf.to_string(index=False))

# 4. ROC curve for the Poor label (one line per model, AUC in the legend)
plt.figure(figsize=(6, 5))
for name in models:
    fpr, tpr, _ = roc_curve(is_poor, probs[name])
    auc = roc_auc_score(is_poor, probs[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guessing (AUC 0.50)")
plt.xlabel("False alarm rate")
plt.ylabel("Poor days caught (recall)")
plt.title("ROC curve: catching Poor days")
plt.legend()
plt.tight_layout()
plt.savefig("docs/roc_curve.png", dpi=150)
plt.close()

# 5. Confusion matrices (rows = what really happened, columns = what the model said)
cm_text = ""
for name, m in models.items():
    cm = confusion_matrix(y_test, m.predict(X_test), labels=CLASSES)
    cm_df = pd.DataFrame(cm, columns=["Pred " + c for c in CLASSES])
    cm_df.insert(0, "Actual", CLASSES)
    cm_text += f"### {name}\n\n" + md_table(cm_df) + "\n"

# 6. Cross-validation on the TRAIN pile only, in time order, with the spread
cv = TimeSeriesSplit(n_splits=5)
cv_rows = []
for name, m in models.items():
    r = cross_validate(clone(m), X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    cv_rows.append({
        "model": name,
        "accuracy_%": f"{r['test_accuracy'].mean()*100:.1f} +/- {r['test_accuracy'].std()*100:.1f}",
        "f1_macro": f"{r['test_f1_macro'].mean():.3f} +/- {r['test_f1_macro'].std():.3f}",
    })
cv_table = pd.DataFrame(cv_rows)
print("\n=== 5-FOLD CROSS-VALIDATION ON TRAIN (mean +/- spread) ===")
print(cv_table.to_string(index=False))

# 7. COST MATRIX and threshold
COST_MISSED_POOR = 5
COST_FALSE_ALARM = 1
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

def cost_at(p, actual, th):
    warn = np.asarray(p) >= th
    actual = np.asarray(actual)
    missed = int((~warn & actual).sum())
    alarms = int((warn & ~actual).sum())
    return missed * COST_MISSED_POOR + alarms * COST_FALSE_ALARM, missed, alarms

# Choose the threshold from predictions made on the TRAIN pile only (5 time-ordered rounds).
# The test pile is NOT used to choose it.
oof_p = {name: [] for name in models}
oof_poor = []
for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_train):
    oof_poor.extend((y_train.iloc[va_idx] == "Poor").values)
    for name, m in models.items():
        f = clone(m).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        if "Poor" in f.classes_:
            p = f.predict_proba(X_train.iloc[va_idx])[:, list(f.classes_).index("Poor")]
        else:
            p = np.zeros(len(va_idx))
        oof_p[name].extend(p)
oof_poor = np.array(oof_poor)
print(f"\nThreshold chosen on {len(oof_poor)} train hours, {oof_poor.sum()} of them Poor")

cost_rows = []
for name in models:
    best_th = min(THRESHOLDS, key=lambda t: (cost_at(oof_p[name], oof_poor, t)[0], -t))
    c50, miss50, al50 = cost_at(probs[name], is_poor, 0.50)
    cb, missb, alb = cost_at(probs[name], is_poor, best_th)
    cost_rows.append({
        "model": name, "chosen_threshold": best_th,
        "missed_at_0.5": miss50, "alarms_at_0.5": al50, "cost_at_0.5": c50,
        "missed_at_chosen": missb, "alarms_at_chosen": alb, "cost_at_chosen": cb,
    })
cost_table = pd.DataFrame(cost_rows)
print("\n=== COST ON THE TEST PILE (missed Poor = 5, false alarm = 1) ===")
print(cost_table.to_string(index=False))

# 8. Write everything into one markdown file
md = "# Part E: Evaluation\n\n"
md += "## Split\nBy time: older 80% = train, newest 20% = test, with a 24-row gap (each answer looks 24 hours ahead).\n\n"
md += f"## 1. Test results (baseline: always guess {base_test.idxmax()} = {baseline}%)\n\n" + md_table(perf)
md += "\n## 2. Why these metrics suit this user\n"
md += (f"- **Poor recall:** the user is a parent, school or outdoor worker. A Poor day with no warning is the worst mistake. "
       f"Recall is the share of real Poor hours the model caught.\n"
       f"- **Poor precision:** if the model warns all the time, recall looks great but nobody trusts the warnings. "
       f"Precision is the share of warnings that were right.\n"
       f"- **Macro F1:** Poor is just {base_test.get('Poor', 0)*100:.1f}% of test hours, so accuracy can look fine while Poor is ignored. "
       f"Macro F1 counts Good, Moderate and Poor equally.\n\n")
md += "## 3. ROC curve and AUC (Poor vs the rest)\n\n![ROC curve](roc_curve.png)\n\n"
md += "## 4. Confusion matrices (rows = actual, columns = predicted)\n\n" + cm_text
md += "## 5. Cross-validation on the train pile (5 folds, time order, mean +/- spread)\n\n" + md_table(cv_table)
md += "\n## 6. Cost matrix and operating threshold\n\n"
md += md_table(pd.DataFrame({"": ["Poor day, warned", "Poor day, not warned", "Not Poor, warned", "Not Poor, not warned"],
                             "cost": [0, COST_MISSED_POOR, COST_FALSE_ALARM, 0]}))
md += ("\nThe costs are our own choice: a missed Poor day is worse than a false alarm. "
       "The threshold was chosen on a validation slice from the end of the train pile, not on the test pile. "
       "The default 0.5 was not kept.\n\n")
md += md_table(cost_table)

with open("docs/evaluation_results.md", "w", encoding="utf-8") as f:
    f.write(md)
print("\nSaved docs/evaluation_results.md and docs/roc_curve.png")