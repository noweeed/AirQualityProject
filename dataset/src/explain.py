import os
import joblib
import pandas as pd
import numpy as np

# Plain-English human labels for the 6 technical sensor codes
FEATURE_NAMES_HUMAN = {
    "CO(GT)": "Carbon Monoxide (CO)",
    "C6H6(GT)": "Benzene (C6H6)",
    "PT08.S2(NMHC)": "Hydrocarbon Sensor (PT08.S2)",
    "NOx(GT)": "Nitrogen Oxides (NOx)",
    "NO2(GT)": "Nitrogen Dioxide (NO2)",
    "PT08.S5(O3)": "Ozone Sensor (PT08.S5)"
}

# Physical units for clear explanation
FEATURE_UNITS = {
    "CO(GT)": "mg/m³",
    "C6H6(GT)": "µg/m³",
    "PT08.S2(NMHC)": "sensor signal",
    "NOx(GT)": "ppb",
    "NO2(GT)": "µg/m³",
    "PT08.S5(O3)": "sensor signal"
}

def load_models():
    """Load the decision tree pipeline and test data."""
    tree_pipe = joblib.load("models/decision_tree.joblib")
    return tree_pipe

def explain_decision_tree_path(pipeline, input_row):
    """
    Extracts the exact decision branch followed for a single input record
    and translates it into plain English (No 'node 14' or 'gini' jargon).
    """
    # 1. Pipeline has imputer and model
    imputer = pipeline.named_steps["fill"]
    tree = pipeline.named_steps["model"]

    # Transform input row through imputer with preserved column names
    feature_cols = list(input_row.index)
    X_df = pd.DataFrame([input_row.values], columns=feature_cols)
    X_val = imputer.transform(X_df)
    
    # 2. Get the prediction and probability
    pred_class = tree.predict(X_val)[0]
    pred_prob = tree.predict_proba(X_val)[0]
    classes = list(tree.classes_)
    confidence = round(pred_prob[classes.index(pred_class)] * 100, 1)

    # 3. Extract the decision path nodes
    tree_obj = tree.tree_
    node_indicator = tree.decision_path(X_val)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rules = []
    for node_id in node_index:
        # Stop at the leaf node
        if tree_obj.feature[node_id] < 0:
            continue

        feat_idx = tree_obj.feature[node_id]
        threshold = tree_obj.threshold[node_id]
        feat_name = feature_cols[feat_idx]
        user_name = FEATURE_NAMES_HUMAN.get(feat_name, feat_name)
        unit = FEATURE_UNITS.get(feat_name, "")
        actual_val = X_val[0][feat_idx]

        if actual_val <= threshold:
            rule = f"{user_name} is {actual_val:.1f} {unit} (lower than or equal to {threshold:.1f} {unit})"
        else:
            rule = f"{user_name} is {actual_val:.1f} {unit} (higher than {threshold:.1f} {unit})"
        
        rules.append(rule)

    # 4. Tailored advice depending on user profile
    advisories = {
        "Good": {
            "School": "Safe for all outdoor physical activities and playground recess.",
            "Outdoor Work": "No protective equipment required. Air quality is clean.",
            "Asthma / Sensitive": "Normal outdoor activity is completely safe."
        },
        "Moderate": {
            "School": "Outdoor activities can proceed normally. Monitor children with known severe asthma.",
            "Outdoor Work": "Generally safe. Consider taking breaks if performing heavy physical exertion.",
            "Asthma / Sensitive": "Keep quick-relief inhalers accessible; watch for mild coughing or tightness."
        },
        "Poor": {
            "School": "HIGH RISK: Cancel outdoor athletic matches and keep children indoors for recess.",
            "Outdoor Work": "Wear an N95/FFP2 protective mask and limit prolonged intense physical exertion.",
            "Asthma / Sensitive": "Stay indoors with windows closed. Keep emergency medication ready."
        }
    }

    return {
        "prediction": pred_class,
        "confidence_%": confidence,
        "decision_rules": rules,
        "advisory": advisories.get(pred_class, {})
    }

def run_explanation_demo():
    print("=" * 65)
    print("PART F: EXPLANATION ENGINE (HUMAN-READABLE RULES)")
    print("=" * 65)

    tree_pipe = load_models()
    test = pd.read_csv("dataset/clean/test.csv", index_col=0, parse_dates=True)
    features = ["CO(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)", "PT08.S5(O3)"]

    # Pick 3 example cases: one Good, one Moderate, one Poor from test set
    samples = {}
    for cls in ["Good", "Moderate", "Poor"]:
        matching = test[test["target"] == cls]
        if len(matching) > 0:
            samples[cls] = matching.iloc[0]

    md_output = ["# Part F: Human-Readable Explanations\n"]
    md_output.append("Every prediction comes with its human-understandable reason, "
                     "following the decision tree's exact path without technical jargon.\n\n")

    for cls, row in samples.items():
        sample_features = row[features]
        timestamp = str(row.name)
        result = explain_decision_tree_path(tree_pipe, sample_features)

        print(f"\n--- Case Timestamp: {timestamp} (Actual: {cls}) ---")
        print(f"Predicted Tomorrow: {result['prediction']} ({result['confidence_%']}% confidence)")
        print("Reasoning path:")
        for r in result["decision_rules"]:
            print(f"  • {r}")
        print("Advisories:")
        for role, adv in result["advisory"].items():
            print(f"  [{role}]: {adv}")

        md_output.append(f"## Example: Predicting {result['prediction']} (Timestamp: {timestamp})\n")
        md_output.append(f"**Predicted Category:** `{result['prediction']}` ({result['confidence_%']}% confidence)\n\n")
        md_output.append("**Decision Path (Why the model made this choice):**\n")
        for r in result["decision_rules"]:
            md_output.append(f"- {r}\n")
        md_output.append("\n**Actionable Advisory for Named Users:**\n")
        for role, adv in result["advisory"].items():
            md_output.append(f"- **{role}:** {adv}\n")
        md_output.append("\n---\n\n")

    os.makedirs("docs", exist_ok=True)
    with open("docs/explanation_demo.md", "w", encoding="utf-8") as f:
        f.write("".join(md_output))

    print("\nSaved docs/explanation_demo.md")

if __name__ == "__main__":
    run_explanation_demo()
