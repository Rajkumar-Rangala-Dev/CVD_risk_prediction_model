import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ======================================
# PATCH: scikit-learn pickle compatibility
# (fixes missing _fill_dtype error)
# ======================================
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = None


# ======================================
# Final training feature order (18 features)
# ======================================
EXPECTED_COLS = [
    "male",
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
    "pulse_pressure",
    "chol_per_bmi",
    "age_sysbp"
]


# ======================================
# Load all model artifacts
# ======================================
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE, "model_artifacts")

def load_artifacts():
    imputer = joblib.load(os.path.join(ARTIFACT_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    lr_base = joblib.load(os.path.join(ARTIFACT_DIR, "lr_base.pkl"))
    rf_base = joblib.load(os.path.join(ARTIFACT_DIR, "rf_base.pkl"))
    meta_clf = joblib.load(os.path.join(ARTIFACT_DIR, "meta_clf.pkl"))

    booster = xgb.Booster()
    booster.load_model(os.path.join(ARTIFACT_DIR, "xgb_booster_full.json"))

    with open(os.path.join(ARTIFACT_DIR, "threshold.txt"), "r") as f:
        threshold = float(f.read().strip())

    return imputer, scaler, lr_base, rf_base, booster, meta_clf, threshold


imputer, scaler, lr_base, rf_base, booster, meta_clf, threshold = load_artifacts()


# ======================================
# Streamlit UI
# ======================================
st.set_page_config(page_title="CVD Risk Predictor", layout="centered")
st.title("❤️ 10-Year Coronary Heart Disease Risk Prediction")


# --------------------------------------
# Collect Inputs
# --------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 50)
male = st.sidebar.selectbox("Sex (Male=1, Female=0)", [1, 0])
education = st.sidebar.selectbox("Education Level (1–4)", [1, 2, 3, 4])

currentSmoker_num = st.sidebar.selectbox("Current Smoker?", [0, 1])
cigsPerDay = st.sidebar.number_input("Cigarettes per day", 0, 80, 0)

BPMeds_num = st.sidebar.selectbox("On BP Medication?", [0, 1])
prevalentStroke_num = st.sidebar.selectbox("History of Stroke?", [0, 1])
prevalentHyp_num = st.sidebar.selectbox("Hypertension?", [0, 1])
diabetes_num = st.sidebar.selectbox("Diabetes?", [0, 1])

totChol = st.sidebar.number_input("Total Cholesterol", 100, 400, 180)
sysBP = st.sidebar.number_input("Systolic BP", 90, 240, 120)
diaBP = st.sidebar.number_input("Diastolic BP", 50, 140, 80)
BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
heartRate = st.sidebar.number_input("Heart Rate", 40, 140, 75)
glucose = st.sidebar.number_input("Glucose", 50, 300, 90)


# ======================================
# Build input_df
# ======================================
input_df = pd.DataFrame([{
    "male": male,
    "age": age,
    "education": education,
    "currentSmoker": currentSmoker_num,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds_num,
    "prevalentStroke": prevalentStroke_num,
    "prevalentHyp": prevalentHyp_num,
    "diabetes": diabetes_num,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose,
}])

# Add engineered features exactly as training did
input_df["pulse_pressure"] = input_df["sysBP"] - input_df["diaBP"]
input_df["chol_per_bmi"] = input_df["totChol"] / (input_df["BMI"] + 1e-9)
input_df["age_sysbp"] = input_df["age"] * input_df["sysBP"]

# Reorder columns
input_df = input_df[EXPECTED_COLS]


# ======================================
# Preprocessing
# ======================================
X_imp = pd.DataFrame(imputer.transform(input_df), columns=EXPECTED_COLS)
X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=EXPECTED_COLS)

# Base model probabilities
p_lr = lr_base.predict_proba(X_scaled)[0, 1]
p_rf = rf_base.predict_proba(X_scaled)[0, 1]

# XGBoost prob
dmat = xgb.DMatrix(X_imp.values, feature_names=EXPECTED_COLS)
p_xgb = booster.predict(dmat)[0]

# Meta model (stacking)
meta_input = np.array([[p_lr, p_rf, p_xgb]])
p_final = meta_clf.predict_proba(meta_input)[0, 1]


# ======================================
# Display prediction
# ======================================
st.subheader("🩺 Predicted 10-Year CHD Risk")
st.metric("Risk Probability", f"{p_final*100:.2f}%")
st.write(f"Decision Threshold: **{threshold:.3f}**")

risk = (
    "🟢 Low Risk" if p_final < 0.10 else
    "🟡 Moderate Risk" if p_final < 0.20 else
    "🔴 High Risk"
)
st.write(f"Risk Category: **{risk}**")


# ======================================
# SHAP Explainability
# ======================================
if st.checkbox("Show SHAP Explanation (XGBoost)"):

    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_imp)

    shap_vec = shap_vals[0] if shap_vals.ndim == 2 else shap_vals[0, :, 0]

    shap_df = pd.DataFrame({
        "feature": EXPECTED_COLS,
        "shap_value": shap_vec
    }).sort_values("shap_value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(shap_df["feature"], shap_df["shap_value"])
    ax.invert_yaxis()
    st.pyplot(fig)

    st.dataframe(shap_df)


# Debug information
with st.expander("Debug Info"):
    st.write("Columns sent to model:", EXPECTED_COLS)
    st.write("Input DF:", input_df)
    st.write("Imputed DF:", X_imp)
