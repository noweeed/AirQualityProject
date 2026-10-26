# Project 14 — Air Quality Health Advisory

A data mining project that predicts tomorrow's air quality category (Good, Moderate, or Poor) from multi-sensor chemical readings and provides specific, human-understandable health advisories for schools, outdoor workers, and individuals with asthma.

---

## Team Members

- **Muhammad Naveed Sajjad** (FA23-BSE-129)
- **Arham Wasim** (FA23-BSE-202)
- **Muneeb Ahmed** (FA23-BSE-145)

**Course:** DSC306 Data Mining (Lectures 1–14)  
---

## What the Product Predicts

1. **Task:** Multiclass classification — forecasting which air quality category tomorrow will fall into:
   - **Good:** NO2 ≤ 100 µg/m³ (Clean, healthy air).
   - **Moderate:** 101 to 200 µg/m³ (Acceptable air, monitor sensitive groups).
   - **Poor:** > 200 µg/m³ (Unhealthy/toxic air, triggers warnings).
2. **Majority-Class Baseline:** **63.1%** (always guessing Moderate).
3. **Core Feature:** Every forecast comes with a plain-English explanation (decision rules, nearest past cases, or probability signals) and actionable advice tailored to schools, outdoor workers, or parents of asthmatic children.

---

## How to Install and Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Data Pipeline (Reproducibility)
To regenerate the datasets, audits, and models from scratch:
```bash
python dataset/src/data_audit.py    # Audits raw data and outputs docs/audit_report.md
python dataset/src/data_clean.py    # Repairs -200 faults and outputs docs/before_after.md
python dataset/src/validate.py      # Verifies 12 data quality assertions
python dataset/src/make_target.py   # Builds tomorrow's 24-hour ahead classification target
python dataset/src/preprocess.py    # Splits train/test by time and compares PCA vs Selection
python dataset/src/models.py        # Tunes and trains Decision Tree, Naive Bayes, and k-NN
python dataset/src/evaluate.py      # Evaluates test metrics, ROC curve, and cost matrix
python dataset/src/explain.py       # Generates human-readable decision path explanations
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to interact with the 6 screens.

---

## What Each File Does

### Application
- `app.py`: The main Streamlit web application featuring all 6 project screens:
  1. **Data & Quality Audit:** Shows the raw sensor flaw audit, before-and-after repair tables, and clean CSV download.
  2. **Model Performance:** Model comparison table with baseline, the -200 challenge proof, ROC curve, and confusion matrices.
  3. **Single Prediction & Advisory:** Live prediction tool where the user selects their role, inputs readings, and gets the prediction, explanation, and tailored advice.
  4. **Batch Scoring:** Upload a CSV of multiple cases, view filterable results, and download predictions.
  5. **Settings & Cost Matrix:** Move the decision threshold slider (1% to 100%) and edit penalties to see live consequences.
  6. **30-Day Forecast Accuracy:** Domain-specific screen showing what the model predicted against what actually happened over the last 30 operational days.

### Scripts (`dataset/src/`)
- `dataset/src/data_audit.py`: Checks raw data for 8 distinct faults (including -200 sentinels, missing readings, and sudden jumps) and saves `docs/audit_report.md`.
- `dataset/src/data_clean.py`: Replaces -200 with NaN, drops empty NMHC column, repairs short gaps (≤3 hours) via linear time interpolation, tracks provenance, and outputs `docs/before_after.md` and `dataset/clean/airquality_clean.csv`.
- `dataset/src/validate.py`: Contains 12 assertion checks ensuring zero -200 values remain, physical sensor limits hold, and timestamps are sorted without duplicates.
- `dataset/src/make_target.py`: Creates tomorrow's target category by shifting NO2 by 24 hours and binning into Good, Moderate, and Poor based on European CAQI standards.
- `dataset/src/preprocess.py`: Performs a time-based train/test split with a 24-hour embargo gap, justifies rescaling decisions, compares PCA vs. Feature Selection on a Decision Tree, and writes `docs/part_c_results.md`.
- `dataset/src/models.py`: Uses 5-fold `TimeSeriesSplit` cross-validation with grid search on the training set to tune hyperparameters for Decision Tree, Naive Bayes, and k-NN, then tests the impact of raw -200 on all three.
- `dataset/src/evaluate.py`: Generates confusion matrices, ROC curves with AUC for Poor air quality, and tests the cost matrix with threshold tuning.
- `dataset/src/explain.py`: Traverses the Decision Tree's path for any case and outputs plain-English rules and tailored role advisories.
- `dataset/src/md_table.py`: Helper script to format clean Markdown tables by hand without external packages.

### Documentation (`docs/`)
- `docs/data_dictionary.md`: Written data dictionary for all attributes, measurement types, plausible ranges, and excluded temporal leakage columns.
- `docs/audit_report.md`: Markdown table of the fault audit.
- `docs/before_after.md`: Numeric before-and-after table demonstrating the effect of repairing -200 sensor faults.
- `docs/part_c_results.md`: Documentation of train/test time splits, baselines, rescaling justifications, and PCA vs Selection comparisons.
- `docs/models_results.md`: Cross-validation scores, final test scores, and the -200 challenge proof.
- `docs/evaluation_results.md`: Complete evaluation report with confusion matrices, cost matrices, and metric justifications.
- `docs/explanation_demo.md`: Real test cases illustrating human-readable explanation paths.
- `docs/roc_curve.png`: ROC plot comparing all three models for Poor air quality.

### Data Directories
- `dataset/raw/`: Contains the original untouched `AirQualityUCI.xlsx` dataset.
- `dataset/clean/`: Contains cleaned data files (`airquality_clean.csv`, `airquality_model.csv`, `train.csv`, `test.csv`).
- `models/`: Saved joblib model pipelines (`decision_tree.joblib`, `naive_bayes.joblib`, `nearest_neighbour.joblib`).
