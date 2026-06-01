# Diabetes Diagnostic Prediction — Patient Prioritization Model

A machine learning pipeline that predicts diabetes risk and ranks patients for targeted outreach, built for clinics with limited staff capacity.

## The Problem

Clinics can't follow up with every patient. Calling a non-diabetic patient isn't just a wrong prediction — it's a wasted call slot that could have reached someone who actually needs intervention. This project reframes diabetes classification as a **ranking and prioritization problem**, optimizing for precision in the contacted cohort rather than overall accuracy.

## Approach

- **Primary Model:** Logistic Regression — chosen for interpretability so clinicians can see exactly which features drive each prediction
- **Key Metric:** Precision at Top-N — "if we call 20 patients this week, how many will actually be diabetic?"
- **Output:** A probability-ranked patient list; care coordinators pick how far down the list to go based on that week's staffing

## Dataset

768 patients, 11 features (Pima Indians Diabetes dataset, modified with additional columns for `ActivityLevel` and `Gender`).

| Feature | Type | Notes |
|---|---|---|
| Glucose | Lab / Objective | Primary diagnostic marker |
| BloodPressure | Lab / Objective | Diastolic BP |
| ThicknessSkin | Lab / Objective | Triceps skinfold thickness |
| InsulinLevel | Lab / Objective | 2-hour serum insulin |
| BMI | Lab / Objective | Body mass index |
| DPF | Derived | Diabetes pedigree function |
| Pregnancies | Admin | Number of pregnancies |
| Age | Admin | Patient age |
| ActivityLevel | Self-reported | Low / Medium / High |
| Gender | Admin | M / F |
| HaveDiabetes | Target | 0 = No, 1 = Yes (34.9% prevalence) |

## Data Cleaning

- **Implausible zeros → NaN:** Glucose, BloodPressure, ThicknessSkin, InsulinLevel, and BMI cannot biologically be zero — these are missing values in disguise. Replaced with NaN and imputed using column median.
- **Cross-column contradiction:** 2 male patients had Pregnancies > 0 — a data entry error. Set to 0.
- **No leakage:** All imputation happens inside sklearn `Pipeline`, fitted on training data only.

## Pipeline Architecture

```
Raw Data
  → zeros_to_nan (rule-based, no fit)
  → add_interactions (Age×BMI, Glucose×BMI, Preg/Age ratio)
  → ColumnTransformer
      ├── Numeric: MedianImputer → StandardScaler
      └── Categorical: ModeImputer → OneHotEncoder
  → Model (LogisticRegression or RandomForestClassifier)
```

All transformations are fitted exclusively on training folds — the test set never influences any preprocessing parameter.

## Results

| Metric | Target | Result | Status |
|---|---|---|---|
| PR-AUC | ≥ 0.65 | 0.656 | ✅ Met |
| ROC-AUC | ≥ 0.75 | 0.807 | ✅ Met |
| Precision at Top-20 | ≥ 0.70 | 0.700 | ✅ Met |
| Precision at Threshold | ≥ 0.75 | 0.634 | ❌ Not met on test |

### Top-N Precision (Logistic Regression)

| Clinic Capacity (N) | True Diabetics Found | Precision | Lift over Random |
|---|---|---|---|
| 5 | 4 | 80.0% | 2.28x |
| 10 | 7 | 70.0% | 2.00x |
| 20 | 14 | 70.0% | 2.00x |
| 30 | 22 | 73.3% | 2.09x |
| 50 | 32 | 64.0% | 1.83x |

At every capacity level, the model outperforms random outreach by 1.8–2.3x.

## Feature Importance

SHAP and permutation importance confirm that the model's top drivers align with clinical knowledge:

1. **Glucose** — strongest predictor (primary diagnostic marker for diabetes)
2. **BMI** — obesity drives insulin resistance
3. **Pregnancies** — gestational diabetes history is a major T2D risk factor
4. **DPF** — genetic/family history burden
5. **Age** — beta-cell function declines with age

## How It Would Be Used

1. Score all patients monthly using the fitted LR pipeline
2. Sort by predicted probability (descending)
3. Hand the ranked list to care coordinators
4. They cut the list based on available call capacity that week
5. Track outcomes to monitor whether precision holds over time

## Limitations

- **Predicts diabetes status, not care gaps** — a patient already on metformin with controlled HbA1c shouldn't be a priority contact, but this model can't tell the difference
- **No HbA1c, medications, or utilization data** — these would be the strongest signals for identifying unmanaged patients
- **Single dataset** — performance may not generalize to a different patient population without prospective validation
- **Fairness not audited** — precision may differ across demographic subgroups

## Tech Stack

- Python, pandas, NumPy, scikit-learn
- SHAP for model explainability
- Matplotlib, Seaborn for visualization
- Jupyter Notebook

## Repository Structure

```
├── diabetes_prediction_kundan_kumar.ipynb   # Full notebook with code + narrative
├── diabetes.xlsx                            # Dataset
├── README.md                                # This file
```

## Author

**Kundan Kumar**
MS in Data Science & Business Analytics, Wayne State University

---

*Built as a take-home assessment demonstrating end-to-end ML pipeline design with a focus on clinical interpretability and operational deployment.*
