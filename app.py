import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# SimpleImputer BACKWARD COMPATIBILITY SHIM (CRITICAL)
# -------------------------------------------------------------
# Many older scikit-learn versions saved SimpleImputer with internal
# attributes that no longer exist in sklearn >= 1.5 (Cloud default).
from sklearn.impute import SimpleImputer

if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = None
# -------------------------------------------------------------


# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="10-Year CVD Risk Predictor", layout="wide")
st.title("❤️ 10-Year Coronary Heart Disease Risk — Stacked Ensemble Model")


# ----------------------------
# Artifact loading utilities
# ----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE, "model_artifacts")

REQUIRED = [
    "imputer.pkl",
    "scaler.pkl",
    "lr_base.pkl",
    "rf_base.pkl",
    "meta_clf.pkl",
    "xgb_booster_full.json",
    "threshold.txt",
]


def load_artifacts():
    """Load all model artifacts safely."""
    loaded = {}
    missing = []

    for fn in REQUIRED:
        path = os.path.join(ARTIFACT_DIR, fn)
        if os.path.exists(path):
            loaded[fn] = path
        else:
            missing.append(fn)

    if missing:
        st.error(f"Missing required artifacts: {missing}")
        st.stop()

    # Load objects
    imputer = joblib.load(loaded["imputer.pkl"])
    scaler = joblib.load(loaded["scaler.pkl"])
    lr_base = joblib.load(loaded["lr_base.pkl"])
    rf_base = joblib.load(loaded["rf_base.pkl"])
    meta_clf = joblib.load(loaded["meta_clf.pkl"])

    # Load XGBoost Booster (2.x compatible)
    booster = xgb.Booster()
    booster.load_model(loaded["xgb_booster_full.json"])

    # Load threshold
    with open(loaded["threshold.txt"], "r") as f:
        threshold = float(f.read().strip())

    return imputer, scaler, lr_base, rf_base, booster, meta_clf, threshold


# Load everything up front
imputer, scaler, lr_base, rf_base, booster, meta_clf, threshold = load_artifacts()


# ----------------------------
# Sidebar Input Form
# ----------------------------
st.sidebar.header("Enter Patient Information")

def sidebar_inputs():
    return pd.DataFrame([{
        "age": st.sidebar.number_input("Age", 20, 100, 50),
        "education": st.sidebar.selectbox("Education Level", [1, 2, 3, 4], index=0),
        "currentSmoker": st.sidebar.selectbox("Current Smoker?", [0, 1], index=0),
        "cigsPerDay": st.sidebar.number_input("Cigarettes per day", 0, 80, 0),
        "BPMeds": st.sidebar.selectbox("On BP Medication?", [0, 1], index=0),
        "prevalentStroke": st.sidebar.selectbox("History of Stroke?", [0, 1], index=0),
        "prevalentHyp": st.sidebar.selectbox("Hypertension?", [0, 1], index=0),
        "diabetes": st.sidebar.selectbox("Diabetes?", [0, 1], index=0),
        "totChol": st.sidebar.number_input("Total Cholesterol", 100, 400, 180),
        "sysBP": st.sidebar.number_input("Systolic BP", 90, 240, 120),
        "diaBP": st.sidebar.number_input("Diastolic BP", 50, 140, 80),
        "BMI": st.sidebar.number_input("BMI", 10.0, 60.0, 25.0),
        "heartRate": st.sidebar.number_input("Heart Rate", 40, 140, 75),
        "glucose": st.sidebar.number_input("Glucose", 50, 300, 90),
        "male": st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1], index=1),
    }])


input_df = sidebar_inputs()

# ---------------------------------------------------------
# TRAINING FEATURE ENGINEERING RECONSTRUCTION
# ---------------------------------------------------------
input_df["age_sysbp"] = input_df["age"] * input_df["sysBP"]
input_df["chol_per_bmi"] = input_df["totChol"] / input_df["BMI"]
input_df["pulse_pressure"] = input_df["sysBP"] - input_df["diaBP"]

# ORDER MUST MATCH TRAINING EXACTLY
expected_cols = [
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "male",
    "age_sysbp",
    "chol_per_bmi",
    "pulse_pressure"
]

# Ensure ALL expected columns exist
missing_cols = [c for c in expected_cols if c not in input_df.columns]
if missing_cols:
    st.error(f"Missing columns in input: {missing_cols}")
    st.stop()

# Reorder
input_df = input_df[expected_cols]


# ----------------------------
# RUN PREDICTION
# ----------------------------
st.subheader("📊 Prediction Result")

# Impute missing values
X_imp = pd.DataFrame(
    imputer.transform(input_df),
    columns=input_df.columns
)

# Scale input
X_scaled = scaler.transform(X_imp)

# Base model outputs
p_lr = lr_base.predict_proba(X_scaled)[:, 1]
p_rf = rf_base.predict_proba(X_imp)[:, 1]

# XGBoost prediction
dmat = xgb.DMatrix(X_imp.values, feature_names=X_imp.columns.tolist())
p_xgb = booster.predict(dmat)

# Stack features
stack_input = np.column_stack([p_lr, p_rf, p_xgb])

# Final prediction
p_final = meta_clf.predict_proba(stack_input)[0, 1]


# ----------------------------
# Apply Risk Threshold
# ----------------------------
pred_class = 1 if p_final >= threshold else 0

risk_level = (
    "🟢 LOW RISK" if p_final < 0.15 else
    "🟡 MODERATE RISK" if p_final < 0.30 else
    "🔴 HIGH RISK"
)

st.metric("Estimated 10-Year CHD Risk", f"{p_final*100:.2f}%")
st.write(f"**Risk Level:** {risk_level}")
st.write(f"**Decision Threshold:** {threshold:.3f}")


# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
st.subheader("🧠 Model Explainability (SHAP)")

explain = st.checkbox("Show SHAP Explanation")

if explain:
    # XGBoost SHAP
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmat)[0]  # 1 sample

    shap_df = pd.DataFrame({
        "feature": X_imp.columns,
        "shap_value": shap_values
    }).sort_values("shap_value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["feature"], shap_df["shap_value"])
    ax.invert_yaxis()
    ax.set_title("XGBoost SHAP Feature Impact")
    st.pyplot(fig)

    st.dataframe(shap_df)


# ----------------------------
# DEBUG INFO (optional)
# ----------------------------
with st.expander("Debug Info"):
    st.write("Working Directory:", os.getcwd())
    st.write("Files:", os.listdir())
    st.write("Artifacts exist:", os.path.exists(ARTIFACT_DIR))

