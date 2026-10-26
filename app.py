import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)

# Set clean, simple page layout
st.set_page_config(
    page_title="Air Quality Health Advisory",
    page_icon="🌤️",
    layout="wide"
)

# -------------------------------------------------------------
# Constant Definitions & Model Loaders
# -------------------------------------------------------------
FEATURES = ["CO(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "NO2(GT)", "PT08.S5(O3)"]
CLASSES = ["Good", "Moderate", "Poor"]

FEATURE_NAMES = {
    "CO(GT)": "Carbon Monoxide (CO) [mg/m³]",
    "C6H6(GT)": "Benzene (C6H6) [µg/m³]",
    "PT08.S2(NMHC)": "Hydrocarbon Sensor (PT08.S2)",
    "NOx(GT)": "Nitrogen Oxides (NOx) [ppb]",
    "NO2(GT)": "Nitrogen Dioxide (NO2) [µg/m³]",
    "PT08.S5(O3)": "Ozone Sensor (PT08.S5)"
}

@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load("models/decision_tree.joblib"),
        "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
        "Nearest Neighbour": joblib.load("models/nearest_neighbour.joblib")
    }

@st.cache_data
def load_datasets():
    train = pd.read_csv("dataset/clean/train.csv", index_col=0, parse_dates=True)
    test = pd.read_csv("dataset/clean/test.csv", index_col=0, parse_dates=True)
    clean = pd.read_csv("dataset/clean/airquality_clean.csv", index_col=0, parse_dates=True)
    return train, test, clean

@st.cache_data
def load_chosen_thresholds():
    """Reads each model's chosen_threshold from docs/evaluation_results.md."""
    thresholds = {
        "Decision Tree": 0.05,
        "Naive Bayes": 0.50,
        "Nearest Neighbour": 0.30
    }
    if os.path.exists("docs/evaluation_results.md"):
        try:
            with open("docs/evaluation_results.md", "r", encoding="utf-8") as f:
                content = f.read()
            if "## 6. Cost matrix and operating threshold" in content:
                sec = content.split("## 6. Cost matrix and operating threshold")[1]
                lines = [l.strip() for l in sec.strip().split("\n") if l.strip().startswith("|")]
                h_idx = -1
                for idx, line in enumerate(lines):
                    if "chosen_threshold" in line:
                        h_idx = idx
                        break
                if h_idx != -1 and h_idx + 2 < len(lines):
                    headers = [c.strip().lower() for c in lines[h_idx].split("|")[1:-1]]
                    m_col = headers.index("model") if "model" in headers else 0
                    th_col = headers.index("chosen_threshold") if "chosen_threshold" in headers else 1
                    for line in lines[h_idx + 2:]:
                        if not line.startswith("|"):
                            break
                        cols = [c.strip() for c in line.split("|")[1:-1]]
                        if len(cols) > max(m_col, th_col):
                            m_name = cols[m_col].strip().lower()
                            val = float(cols[th_col])
                            for k in list(thresholds.keys()):
                                if k.lower() in m_name or m_name in k.lower():
                                    thresholds[k] = val
        except Exception:
            pass
    return thresholds

MODEL_THRESHOLDS = load_chosen_thresholds()

def get_default_threshold(model_name=None):
    if model_name is None:
        model_name = st.session_state.get("selected_model_name", "Decision Tree")
    return MODEL_THRESHOLDS.get(model_name, 0.05)

def load_challenge_data():
    """Reads the -200 challenge comparison table from docs/models_results.md."""
    if not os.path.exists("docs/models_results.md"):
        return pd.DataFrame()
    with open("docs/models_results.md", "r", encoding="utf-8") as f:
        content = f.read()
    if "## What -200 did to the models" in content:
        sub = content.split("## What -200 did to the models")[1]
        lines = [line.strip() for line in sub.strip().split("\n") if line.strip().startswith("|")]
        if len(lines) >= 3:
            headers = [c.strip() for c in lines[0].split("|")[1:-1]]
            rows = []
            for line in lines[2:]:
                if not line.startswith("|"):
                    break
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) == len(headers):
                    rows.append(cols)
            df = pd.DataFrame(rows, columns=headers)
            df = df.rename(columns={
                "model": "Model",
                "acc_raw": "Acc (Raw -200)",
                "acc_clean": "Acc (Clean)",
                "f1_raw": "F1 (Raw)",
                "f1_clean": "F1 (Clean)"
            })
            df["Acc (Raw -200)"] = df["Acc (Raw -200)"].apply(lambda x: f"{x}%" if not str(x).endswith("%") else str(x))
            df["Acc (Clean)"] = df["Acc (Clean)"].apply(lambda x: f"{x}%" if not str(x).endswith("%") else str(x))
            return df
    return pd.DataFrame()

models = load_models()
train_df, test_df, clean_df = load_datasets()

# Dynamic baseline calculations from train.csv and test.csv
train_base_counts = train_df["target"].value_counts(normalize=True)
test_base_counts = test_df["target"].value_counts(normalize=True)
train_majority_label = train_base_counts.idxmax()
train_baseline_pct = round(train_base_counts.max() * 100, 1)
test_majority_label = test_base_counts.idxmax()
test_baseline_pct = round(test_base_counts.max() * 100, 1)

# -------------------------------------------------------------
# Shared Streamlit Session State for Settings
# -------------------------------------------------------------
DEFAULT_MODEL = "Decision Tree"
DEFAULT_WARNING_LEVEL = get_default_threshold("Decision Tree")

if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = DEFAULT_MODEL
if "poor_threshold" not in st.session_state:
    st.session_state["poor_threshold"] = DEFAULT_WARNING_LEVEL
if "cost_fn" not in st.session_state:
    st.session_state["cost_fn"] = 5
if "cost_fp" not in st.session_state:
    st.session_state["cost_fp"] = 1

# -------------------------------------------------------------
# Navigation Sidebar: 6 Simple Screens
# -------------------------------------------------------------
st.sidebar.title("Air Quality Advisory")
st.sidebar.caption("Data Mining Project")

screens = [
    "1. Data & Quality Audit",
    "2. Model Performance",
    "3. Single Prediction & Advisory",
    "4. Batch Scoring",
    "5. Settings & Cost Matrix",
    "6. 30-Day Forecast Accuracy"
]
selected_screen = st.sidebar.radio("Go to Screen:", screens)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Target:** Tomorrow's Air Quality (Good, Moderate, Poor)\n\n"
    f"**Train Baseline:** {train_baseline_pct}% ({train_majority_label})\n\n"
    f"**Test Majority:** {test_baseline_pct}% ({test_majority_label})\n\n"
    "**Active Model:** " + st.session_state["selected_model_name"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Team Members:**\n"
    "- Muhammad Naveed Sajjad (FA23-BSE-129)\n"
    "- Arham Wasim (FA23-BSE-202)\n"
    "- Muneeb Ahmed (FA23-BSE-145)"
)

# =============================================================
# SCREEN 1: Data & Quality Audit
# =============================================================
if selected_screen == "1. Data & Quality Audit":
    st.header("Screen 1: Data & Data Quality Audit")

    tab1, tab2, tab3 = st.tabs(["1. What Was Wrong (Fault Audit)", "2. How We Fixed It (Before & After)", "3. Clean Dataset"])

    with tab1:
        st.subheader("Sensor Faults Found in the Raw Data")
        st.write(
            "Sensors in the real world break, lose power, or lose signal. "
            "In this dataset, whenever a sensor failed, it recorded -200 instead of leaving an empty cell. "
            "Because -200 looks like a normal negative number, computers do not know it is missing and treat it as real data. "
            "Here are the checks we ran and how many rows were affected:"
        )
        if os.path.exists("docs/audit_report.md"):
            with open("docs/audit_report.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.warning("docs/audit_report.md not found.")

    with tab2:
        st.subheader("Our Cleaning and Repair Process")
        st.write(
            "Here is the step-by-step repair process we carried out:\n"
            "1. **Replaced -200 with NaN:** We converted every -200 into a real missing value so it no longer skews averages.\n"
            "2. **Dropped NMHC(GT):** Over 90% of this column was missing. You cannot safely guess 90% of a column without making up fake data, so we dropped it completely.\n"
            "3. **Repaired Short Gaps (1 to 3 hours):** If a sensor went silent for 1 to 3 hours, we filled it with linear time interpolation (drawing a straight line between the hour before and the hour after).\n"
            "4. **Left Long Gaps Alone:** If a sensor died for days or weeks, we left it blank. Filling long gaps with straight lines would invent fake weather.\n"
            "5. **Provenance Tracking:** We added a companion note column (`_src`) for every sensor recording whether each value was `original`, `interpolated`, or `unrepaired`."
        )

        st.write("**Before and After Numbers (Notice how -200 crushed the raw averages):**")
        if os.path.exists("docs/before_after.md"):
            with open("docs/before_after.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.warning("docs/before_after.md not found.")

    with tab3:
        st.subheader("Clean Repaired Data")
        st.write(
            f"The final dataset has **{len(clean_df):,} hourly records** running from March 2004 to April 2005. "
            "It passed all 12 assertion checks in `validate.py` with zero -200 values remaining."
        )
        csv_data = clean_df.to_csv().encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="airquality_clean.csv",
            mime="text/csv"
        )

# =============================================================
# SCREEN 2: Model Performance
# =============================================================
elif selected_screen == "2. Model Performance":
    st.header("Screen 2: Model Performance Comparison")
    st.write(
        f"Evaluation of three distinct model families on unseen test data ({test_df.index[0].strftime('%b %d')} – {test_df.index[-1].strftime('%b %d, %Y')}). "
        f"The training majority baseline is **{train_baseline_pct}% ({train_majority_label})**, while the test-period majority proportion is **{test_baseline_pct}% ({test_majority_label})**."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("1. Comparison Across Models")
        X_test = test_df[FEATURES]
        y_test = test_df["target"]
        is_poor = (y_test == "Poor").values
        perf_rows = []
        for name, m in models.items():
            pred = m.predict(X_test)
            poor_idx = list(m.classes_).index("Poor")
            poor_probs = m.predict_proba(X_test)[:, poor_idx]
            all_probs = m.predict_proba(X_test)
            
            acc = accuracy_score(y_test, pred) * 100
            f1 = f1_score(y_test, pred, average="macro")
            recall = recall_score(y_test, pred, labels=["Poor"], average=None, zero_division=0)[0]
            prec = precision_score(y_test, pred, labels=["Poor"], average=None, zero_division=0)[0]
            auc_p = roc_auc_score(is_poor, poor_probs)
            auc_m = roc_auc_score(y_test, all_probs, multi_class="ovr", average="macro", labels=CLASSES)
            
            perf_rows.append({
                "Model": name,
                "Train Base": f"{train_baseline_pct}%",
                "Test Base": f"{test_baseline_pct}%",
                "Accuracy": f"{round(acc, 1):.1f}%",
                "Macro F1": f"{round(f1, 3):.3f}",
                "Poor Recall": f"{round(recall, 3):.3f}",
                "Poor Prec": f"{round(prec, 3):.3f}",
                "AUC (Poor)": f"{round(auc_p, 3):.3f}",
                "Multi AUC": f"{round(auc_m, 3):.3f}"
            })
        st.table(pd.DataFrame(perf_rows))

        st.subheader("2. Effect of -200 on Models")
        st.table(load_challenge_data())

    with col2:
        st.subheader("3. ROC Curve (Catching Poor Days)")
        if os.path.exists("docs/roc_curve.png"):
            st.image("docs/roc_curve.png", caption="One-vs-Rest ROC Curve for Poor Air Quality", use_container_width=True)
        else:
            st.info("ROC Curve image not found in docs/roc_curve.png")

    st.subheader("4. Confusion Matrices (Rows: Actual | Columns: Predicted)")
    m_choice = st.selectbox("Select model to view confusion matrix:", list(models.keys()))
    cm_model = models[m_choice]
    preds = cm_model.predict(test_df[FEATURES])
    cm = confusion_matrix(test_df["target"], preds, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in CLASSES], columns=[f"Pred {c}" for c in CLASSES])
    st.table(cm_df)

# =============================================================
# SCREEN 3: Single Prediction & Advisory (Core User Screen)
# =============================================================
elif selected_screen == "3. Single Prediction & Advisory":
    st.header("Screen 3: Single Case Air Quality Advisory")
    st.write(
        "Enter today's sensor readings to forecast tomorrow's air category, "
        "receive tailored advice for your situation, and see the explanation."
    )

    chosen_model_name = st.session_state.get("selected_model_name", DEFAULT_MODEL)
    st.info(f"**Active Forecasting Model:** {chosen_model_name}")

    user_role = st.selectbox(
        "Who is using this advisory?",
        ["School Administrator", "Outdoor Worker", "Parent / Asthmatic Individual"]
    )

    # Presets mapping - exact match to radio options
    PRESETS = {
        "Moderate Day (Average City Air)": {
            "CO(GT)": 2.2, "C6H6(GT)": 9.5, "PT08.S2(NMHC)": 950.0, "NOx(GT)": 250.0, "NO2(GT)": 140.0, "PT08.S5(O3)": 1050.0
        },
        "Clean Day (Low Pollution)": {
            "CO(GT)": 0.8, "C6H6(GT)": 1.4, "PT08.S2(NMHC)": 534.0, "NOx(GT)": 75.0, "NO2(GT)": 64.0, "PT08.S5(O3)": 520.0
        },
        "Poor Day (High Pollution)": {
            "CO(GT)": 3.5, "C6H6(GT)": 15.1, "PT08.S2(NMHC)": 1153.0, "NOx(GT)": 584.0, "NO2(GT)": 283.0, "PT08.S5(O3)": 1667.0
        }
    }

    scenario = st.radio(
        "Quick Example Days:",
        list(PRESETS.keys()),
        horizontal=True,
        key="preset_radio"
    )
    preset = PRESETS[scenario]

    st.subheader("Enter Sensor Readings")
    col1, col2, col3 = st.columns(3)

    # Scoped keys ensure switching presets immediately updates widget values
    with col1:
        co = st.number_input("Carbon Monoxide (CO) [mg/m³]", min_value=0.0, max_value=25.0, value=float(preset["CO(GT)"]), step=0.1, key=f"co_{scenario}")
        c6h6 = st.number_input("Benzene (C6H6) [µg/m³]", min_value=0.0, max_value=100.0, value=float(preset["C6H6(GT)"]), step=0.5, key=f"c6h6_{scenario}")
    with col2:
        pt_nmhc = st.number_input("Hydrocarbon Sensor (PT08.S2)", min_value=0.0, max_value=3000.0, value=float(preset["PT08.S2(NMHC)"]), step=10.0, key=f"pt_nmhc_{scenario}")
        nox = st.number_input("Nitrogen Oxides (NOx) [ppb]", min_value=0.0, max_value=2000.0, value=float(preset["NOx(GT)"]), step=10.0, key=f"nox_{scenario}")
    with col3:
        no2 = st.number_input("Nitrogen Dioxide (NO2) [µg/m³]", min_value=0.0, max_value=500.0, value=float(preset["NO2(GT)"]), step=5.0, key=f"no2_{scenario}")
        pt_o3 = st.number_input("Ozone Sensor (PT08.S5)", min_value=0.0, max_value=3500.0, value=float(preset["PT08.S5(O3)"]), step=10.0, key=f"pt_o3_{scenario}")

    input_data = pd.DataFrame([[co, c6h6, pt_nmhc, nox, no2, pt_o3]], columns=FEATURES)

    if st.button("Forecast Tomorrow's Air Quality", type="primary"):
        # Validate against negative inputs
        if any(v < 0 for v in [co, c6h6, pt_nmhc, nox, no2, pt_o3]):
            st.error("Invalid Input: Sensor readings cannot be negative.")
            st.stop()

        active_pred_model = models[chosen_model_name]
        raw_pred = active_pred_model.predict(input_data)[0]
        probs = active_pred_model.predict_proba(input_data)[0]
        classes = list(active_pred_model.classes_)

        # Apply operating warning level from Screen 5 Settings
        poor_idx = classes.index("Poor")
        poor_prob = probs[poor_idx]
        current_threshold = st.session_state.get("poor_threshold", DEFAULT_WARNING_LEVEL)

        if poor_prob >= current_threshold and raw_pred != "Poor":
            final_pred = "Poor"
            threshold_triggered = True
        else:
            final_pred = raw_pred
            threshold_triggered = False

        conf = round(probs[classes.index(final_pred)] * 100, 1)

        # Color-coded result card
        st.markdown("---")
        st.subheader("Tomorrow's Expected Category")

        if final_pred == "Good":
            st.success(f"### 🟢 GOOD (Clean Air) — {conf}% Confidence (via {chosen_model_name})")
        elif final_pred == "Moderate":
            st.warning(f"### 🟡 MODERATE (Acceptable Air) — {conf}% Confidence (via {chosen_model_name})")
        else:
            st.error(f"### 🔴 POOR (Unhealthy Air) — {conf}% Confidence (via {chosen_model_name})")

        if threshold_triggered:
            st.write("Poor air warning is on.")

        # Specific user recommendation
        st.subheader(f"Specific Recommendation for {user_role}:")
        advisories = {
            "Good": {
                "School Administrator": "Safe for all outdoor physical activities and playground recess.",
                "Outdoor Worker": "No protective equipment required. Air quality is clean.",
                "Parent / Asthmatic Individual": "Normal outdoor activity is completely safe."
            },
            "Moderate": {
                "School Administrator": "Outdoor sports can proceed normally. Keep a watchful eye on children with known severe asthma.",
                "Outdoor Worker": "Generally safe. Consider taking breaks if performing heavy physical exertion.",
                "Parent / Asthmatic Individual": "Keep quick-relief inhalers accessible; monitor for mild coughing or chest tightness."
            },
            "Poor": {
                "School Administrator": "HIGH ALERT: Cancel outdoor athletic matches and keep all children indoors for recess.",
                "Outdoor Worker": "Wear an N95/FFP2 protective mask and reduce intense continuous physical exertion.",
                "Parent / Asthmatic Individual": "Stay indoors with windows closed. Keep emergency rescue medication ready."
            }
        }
        st.info(advisories[final_pred][user_role])

        # Explanation section customized for each model family
        st.subheader("Why the model made this forecast:")

        if chosen_model_name == "Decision Tree":
            # Extract tree decision path and simplify into clear physical limits
            tree_pipe = active_pred_model
            imputer = tree_pipe.named_steps["fill"]
            tree = tree_pipe.named_steps["model"]
            X_val = imputer.transform(input_data)

            tree_obj = tree.tree_
            node_indicator = tree.decision_path(X_val)
            node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

            st.write("**Decision Rules Followed:**")
            for node_id in node_index:
                if tree_obj.feature[node_id] >= 0:
                    feat_idx = tree_obj.feature[node_id]
                    threshold = tree_obj.threshold[node_id]
                    col_code = FEATURES[feat_idx]
                    friendly_name = FEATURE_NAMES.get(col_code, col_code)
                    val = input_data[col_code].iloc[0]

                    if val > threshold:
                        st.write(f"- **{friendly_name}** is **{val:.1f}**, which is above {threshold:.1f}.")
                    else:
                        st.write(f"- **{friendly_name}** is **{val:.1f}**, which is below or equal to {threshold:.1f}.")

        elif chosen_model_name == "Nearest Neighbour":
            # Find the closest 5 historical training days
            knn_pipe = active_pred_model
            imputer = knn_pipe.named_steps["fill"]
            scaler = knn_pipe.named_steps["scale"]
            knn = knn_pipe.named_steps["model"]

            # Scale input
            X_scaled = scaler.transform(imputer.transform(input_data))
            distances, indices = knn.kneighbors(X_scaled, n_neighbors=5)

            st.write("**Top 5 Most Similar Historical Days in Past Records:**")
            st.caption("k-NN checks past historical days with matching chemical readings and votes on what happened next:")

            history_rows = []
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
                past_date = str(train_df.index[idx].date())
                past_outcome = train_df["target"].iloc[idx]
                past_no2 = train_df["NO2(GT)"].iloc[idx]
                past_co = train_df["CO(GT)"].iloc[idx]
                history_rows.append({
                    "Rank": f"#{rank}",
                    "Past Date": past_date,
                    "Similarity Distance": round(float(dist), 2),
                    "What Happened Next Day": past_outcome,
                    "NO2 on that day (µg/m³)": round(float(past_no2), 1),
                    "CO on that day (mg/m³)": round(float(past_co), 1)
                })

            st.table(pd.DataFrame(history_rows))

        else:
            st.write("**Sensor Signals Compared to Historical Class Means:**")
            nb_pipe = active_pred_model
            nb = nb_pipe.named_steps["model"]
            pred_idx = classes.index(final_pred)
            means = nb.theta_[pred_idx]

            for i, col_code in enumerate(FEATURES):
                val = input_data[col_code].iloc[0]
                expected = means[i]
                friendly_name = FEATURE_NAMES.get(col_code, col_code)
                if val >= expected:
                    st.write(f"- **{friendly_name}:** Reading is **{val:.1f}** (higher than the typical {expected:.1f} on {final_pred} days).")
                else:
                    st.write(f"- **{friendly_name}:** Reading is **{val:.1f}** (lower than the typical {expected:.1f} on {final_pred} days).")

# =============================================================
# SCREEN 4: Batch Scoring
# =============================================================
elif selected_screen == "4. Batch Scoring":
    st.header("Screen 4: Batch Case Scoring")
    st.write("Upload a CSV file containing multiple rows of sensor readings to generate forecasts in bulk.")
    
    active_model_name = st.session_state.get("selected_model_name", DEFAULT_MODEL)
    threshold = st.session_state.get("poor_threshold", DEFAULT_WARNING_LEVEL)
    st.info(f"**Active Forecasting Model:** {active_model_name}")

    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])
    if uploaded_file is not None:
        if uploaded_file.size == 0:
            st.error("Validation Error: Uploaded CSV file is empty. Please upload a valid CSV containing data.")
        else:
            batch_df = None
            try:
                # 1. Try reading without assuming index column
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file)
                # If features not all present, check if first column was index
                if not all(c in batch_df.columns for c in FEATURES):
                    uploaded_file.seek(0)
                    batch_df = pd.read_csv(uploaded_file, index_col=0)
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")
                batch_df = None

            if batch_df is not None:
                missing_cols = [c for c in FEATURES if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing required sensor columns in CSV: {missing_cols}")
                    st.info("Expected columns: " + ", ".join(FEATURES))
                elif len(batch_df) == 0:
                    st.warning("The uploaded CSV contains header columns but 0 data rows.")
                else:
                    # Coerce columns to numeric safely
                    numeric_batch = batch_df[FEATURES].apply(pd.to_numeric, errors='coerce')
                    non_numeric_count = int(numeric_batch.isna().sum().sum())
                    negative_count = int((numeric_batch < 0).sum().sum())
                    
                    if non_numeric_count > 0 or negative_count > 0:
                        st.warning(f"Data Cleaning Note: Replaced {non_numeric_count} non-numeric values and {negative_count} negative entries with median values.")

                    # Replace negatives with NaN so imputer fills them safely
                    clean_inputs = numeric_batch.copy()
                    clean_inputs[clean_inputs < 0] = np.nan

                    active_model = models[active_model_name]
                    raw_preds = active_model.predict(clean_inputs)
                    probs = active_model.predict_proba(clean_inputs)
                    classes = list(active_model.classes_)
                    poor_idx = classes.index("Poor")
                    poor_probs = probs[:, poor_idx]

                    # Apply Screen 5 warning level the same way Screen 3 does
                    final_forecast = np.where(poor_probs >= threshold, "Poor", raw_preds)

                    # Real explanations customized per model family (no probability numbers)
                    explanations = []
                    if active_model_name == "Decision Tree":
                        tree_pipe = active_model
                        imputer = tree_pipe.named_steps["fill"]
                        tree = tree_pipe.named_steps["model"]
                        X_val = imputer.transform(clean_inputs)
                        tree_obj = tree.tree_
                        node_indicator = tree.decision_path(X_val)

                        for i in range(len(clean_inputs)):
                            node_index = node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]]
                            splits = []
                            for node_id in node_index:
                                if tree_obj.feature[node_id] >= 0:
                                    feat_idx = tree_obj.feature[node_id]
                                    th_val = tree_obj.threshold[node_id]
                                    col_code = FEATURES[feat_idx]
                                    actual_val = X_val[i, feat_idx]
                                    if actual_val <= th_val:
                                        splits.append(f"{col_code} <= {th_val:.1f}")
                                    else:
                                        splits.append(f"{col_code} > {th_val:.1f}")
                            top_splits = splits[:3]
                            explanations.append("; ".join(top_splits))

                    elif active_model_name == "Naive Bayes":
                        nb_pipe = active_model
                        nb = nb_pipe.named_steps["model"]
                        for i in range(len(clean_inputs)):
                            f_cat = final_forecast[i]
                            f_idx = classes.index(f_cat)
                            means = nb.theta_[f_idx]
                            signals = []
                            for j, col_code in enumerate(FEATURES):
                                val = clean_inputs.iloc[i][col_code]
                                if val >= means[j]:
                                    signals.append(f"{col_code} high")
                                else:
                                    signals.append(f"{col_code} low")
                            explanations.append("; ".join(signals[:3]))

                    elif active_model_name == "Nearest Neighbour":
                        knn_pipe = active_model
                        imputer = knn_pipe.named_steps["fill"]
                        scaler = knn_pipe.named_steps["scale"]
                        knn = knn_pipe.named_steps["model"]
                        X_scaled = scaler.transform(imputer.transform(clean_inputs))
                        distances, indices = knn.kneighbors(X_scaled, n_neighbors=3)
                        for i in range(len(clean_inputs)):
                            past_items = []
                            for idx in indices[i]:
                                past_date = str(train_df.index[idx].date())
                                past_target = train_df["target"].iloc[idx]
                                past_items.append(f"{past_date} -> {past_target}")
                            explanations.append("3 closest days: " + "; ".join(past_items))

                    results_df = batch_df.copy()
                    results_df["Model_Used"] = active_model_name
                    results_df["Forecast_Category"] = final_forecast
                    results_df["Reason"] = explanations

                    # Keep sorting rows so the most likely Poor days come first, but do not show a P(Poor) column
                    results_df["_sort_key"] = poor_probs
                    results_df = results_df.sort_values(by="_sort_key", ascending=False).drop(columns=["_sort_key"])

                    st.subheader(f"Successfully scored {len(results_df)} records")
                    
                    filter_choice = st.selectbox("Filter results by forecast category:", ["All", "Poor", "Moderate", "Good"])
                    if filter_choice != "All":
                        display_df = results_df[results_df["Forecast_Category"] == filter_choice]
                    else:
                        display_df = results_df

                    st.dataframe(display_df, use_container_width=True)

                    csv_out = results_df.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        "Download Scored Predictions (CSV)",
                        csv_out,
                        "batch_predictions_result.csv",
                        "text/csv"
                    )

# =============================================================
# SCREEN 5: Settings & Cost Matrix
# =============================================================
elif selected_screen == "5. Settings & Cost Matrix":
    st.header("Screen 5: Settings, Cost Matrix & Decision Threshold")
    st.write(
        "Adjust operating thresholds and penalty costs live. "
        "Settings configured here are saved into active session state and applied across Screen 3, Screen 4, and Screen 6."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Active Model Selection")
        model_list = list(models.keys())
        current_idx = model_list.index(st.session_state["selected_model_name"]) if st.session_state["selected_model_name"] in model_list else 0

        selected_model_name = st.selectbox(
            "Choose Active Forecasting Model:",
            model_list,
            index=current_idx,
            key="settings_model_choice"
        )
        st.session_state["selected_model_name"] = selected_model_name
        current_model = models[selected_model_name]

        st.subheader("2. Cost Matrix Definition")
        cost_fn = st.number_input(
            "Cost of Missed Poor Day (False Negative)",
            min_value=1,
            value=int(st.session_state["cost_fn"]),
            step=1,
            key="settings_cost_fn"
        )
        st.session_state["cost_fn"] = cost_fn

        cost_fp = st.number_input(
            "Cost of False Alarm (False Positive)",
            min_value=1,
            value=int(st.session_state["cost_fp"]),
            step=1,
            key="settings_cost_fp"
        )
        st.session_state["cost_fp"] = cost_fp

    with col2:
        st.subheader("3. Threshold")
        def on_warning_level_change():
            st.session_state["poor_threshold"] = float(st.session_state["settings_th_slider"])

        th = st.slider(
            "Warning level",
            min_value=0.01,
            max_value=1.00,
            value=float(st.session_state["poor_threshold"]),
            step=0.01,
            format="%.2f",
            key="settings_th_slider",
            on_change=on_warning_level_change
        )
        st.session_state["poor_threshold"] = th
        st.caption(f"If the model is at least **{int(th*100)}%** sure tomorrow will be Poor, it sounds the alarm.")

    # Compute live cost evaluation on test set
    y_test = test_df["target"]
    is_poor = (y_test == "Poor").values
    poor_idx = list(current_model.classes_).index("Poor")
    poor_probs = current_model.predict_proba(test_df[FEATURES])[:, poor_idx]

    warn = poor_probs >= th
    missed = int((~warn & is_poor).sum())
    alarms = int((warn & ~is_poor).sum())
    caught = int((warn & is_poor).sum())
    total_poor = int(is_poor.sum())
    total_cost = (missed * cost_fn) + (alarms * cost_fp)
    recall = round((caught / total_poor) * 100, 1) if total_poor > 0 else 0

    st.markdown("---")
    st.subheader(f"Live Impact on Test Period ({len(test_df)} hours, {total_poor} actual Poor hours):")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Poor days caught", f"{caught} / {total_poor}", f"{recall}% Recall")
    c2.metric("Poor days missed", f"{missed}", f"Penalty: {missed * cost_fn}")
    c3.metric("False alarms", f"{alarms}", f"Penalty: {alarms * cost_fp}")
    c4.metric("Total cost", f"{total_cost}")

# =============================================================
# SCREEN 6: 30-Day Forecast Accuracy
# =============================================================
elif selected_screen == "6. 30-Day Forecast Accuracy":
    st.header("Screen 6: 30-Day Forecast Accuracy")

    active_model_name = st.session_state.get("selected_model_name", DEFAULT_MODEL)
    model = models[active_model_name]
    th = st.session_state.get("poor_threshold", DEFAULT_WARNING_LEVEL)
    st.info(f"**Active Model:** {active_model_name}")

    # Extract the last 30 daily readings from the test set (12:00 PM daily readings)
    test_daily = test_df[test_df.index.hour == 12].tail(30).copy()
    start_date = test_daily.index[0].strftime("%B %d, %Y")
    end_date = test_daily.index[-1].strftime("%B %d, %Y")

    st.write(
        f"This screen shows the last 30 days ({start_date} to {end_date}). "
        "For each day at 12:00 PM, the model made a 24-hour forecast using that day's sensor readings. "
        "Here is the day-by-day comparison between what was predicted and what actually happened."
    )

    # Apply Screen 5 threshold to predictions
    raw_preds = model.predict(test_daily[FEATURES])
    poor_idx = list(model.classes_).index("Poor")
    poor_probs = model.predict_proba(test_daily[FEATURES])[:, poor_idx]
    preds = np.where(poor_probs >= th, "Poor", raw_preds)

    test_daily["Predicted"] = preds
    test_daily["Actual"] = test_daily["target"]
    test_daily["Match"] = test_daily["Predicted"] == test_daily["Actual"]

    correct_count = int(test_daily["Match"].sum())
    accuracy_30d = round((correct_count / len(test_daily)) * 100, 1)

    base_30d = test_daily["Actual"].value_counts(normalize=True)
    majority_30d = base_30d.idxmax()
    baseline_30d_pct = round(base_30d.max() * 100, 1)

    # 30-Day Scorecard
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("30-Day Overall Accuracy", f"{accuracy_30d}%", f"{correct_count} of {len(test_daily)} days correct")
    col2.metric("Baseline (Biggest Group)", f"{baseline_30d_pct}%", f"Always guess {majority_30d}")
    col3.metric("Missed Poor Days", f"{int(((test_daily['Actual'] == 'Poor') & (test_daily['Predicted'] != 'Poor')).sum())} days")
    col4.metric("False Alarms", f"{int(((test_daily['Predicted'] == 'Poor') & (test_daily['Actual'] != 'Poor')).sum())} days")

    st.subheader("Day-by-Day Historical Log (Last 30 Days):")

    def format_status(row):
        if row["Match"]:
            return "✅ Correct"
        elif row["Actual"] == "Poor" and row["Predicted"] != "Poor":
            return "❌ MISSED POOR DAY"
        else:
            return "⚠️ Category Mismatch"

    test_daily["Status"] = test_daily.apply(format_status, axis=1)

    display_table = test_daily[["Actual", "Predicted", "Status", "NO2(GT)", "CO(GT)"]].copy()
    display_table.columns = ["Actual Reality", "Model Forecast", "Result Status", "NO2 (µg/m³)", "CO (mg/m³)"]
    st.dataframe(display_table, use_container_width=True)
