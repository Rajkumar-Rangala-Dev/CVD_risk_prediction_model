import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
import xgboost as xgb
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CVD Risk Prediction",
    layout="centered",
    page_icon="❤️"
)

st.title("❤️ Cardiovascular Disease Risk Prediction")
st.write("This app predicts the **10-year CVD risk** using Logistic Regression, Random Forest, and XGBoost, "
         "and combines them using a safe, stable ensemble method compatible with Streamlit Cloud.")

ART = "model_artifacts"

# ============================================
# LOAD ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    imputer = joblib.load(os.path.join(ART, "imputer.pkl"))
    scaler = joblib.load(os.path.join(ART, "scaler.pkl"))
    lr_base = joblib.load(os.path.join(ART, "lr_base.pkl"))
    rf_base = joblib.load(os.path.join(ART, "rf_base.pkl"))

    # Load XGBoost booster manually
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(ART, "xgb_booster_full.json"))

    with open(os.path.join(ART, "threshold.txt")) as f:
        threshold = float(f.read().strip())

    return imputer, scaler, lr_base, rf_base, xgb_booster, threshold

imputer, scaler, lr_base, rf_base, xgb_booster, threshold = load_artifacts()

# Feature order expected by models
FEATURES = [
    "male","age","education","currentSmoker","cigsPerDay","BPMeds",
    "prevalentStroke","prevalentHyp","diabetes","totChol","sysBP",
    "diaBP","BMI","heartRate","glucose","pulse_pressure","chol_per_bmi","age_sysbp"
]

# ============================================
# SIDEBAR INPUTS
# ============================================
st.sidebar.header("Patient Information")

def num(label, min_, max_, default):
    return st.sidebar.number_input(label, min_value=min_, max_value=max_, value=default)

def cat(label):
    return st.sidebar.selectbox(label, [0, 1])

input_df = pd.DataFrame([{
    "male": cat("Male? (1 yes / 0 no)"),
    "age": num("Age", 20, 90, 50),
    "education": num("Education Level", 1, 4, 1),
    "currentSmoker": cat("Current Smoker?"),
    "cigsPerDay": num("Cigarettes Per Day", 0, 60, 0),
    "BPMeds": cat("On BP Medication?"),
    "prevalentStroke": cat("Prior Stroke?"),
    "prevalentHyp": cat("Hypertension?"),
    "diabetes": cat("Diabetes?"),
    "totChol": num("Total Cholesterol", 100, 600, 200),
    "sysBP": num("Systolic BP", 80, 250, 120),
    "diaBP": num("Diastolic BP", 40, 150, 80),
    "BMI": num("BMI", 10.0, 60.0, 25.0),
    "heartRate": num("Heart Rate", 40, 150, 75),
    "glucose": num("Glucose", 40, 400, 90),
    # Derived features
    "pulse_pressure": 0.0,
    "chol_per_bmi": 0.0,
    "age_sysbp": 0.0
}])

# Derived features
input_df["pulse_pressure"] = input_df["sysBP"] - input_df["diaBP"]
input_df["chol_per_bmi"] = input_df["totChol"] / input_df["BMI"]
input_df["age_sysbp"] = input_df["age"] * input_df["sysBP"]

# ============================================
# IMPUTE + SCALE
# ============================================
X_imp = pd.DataFrame(imputer.transform(input_df[FEATURES]), columns=FEATURES)
X_scaled = scaler.transform(X_imp)

# ============================================
# MODEL PREDICTIONS
# ============================================
# 1) Logistic Regression
lr_p = lr_base.predict_proba(X_scaled)[0, 1]

# 2) Random Forest
rf_p = rf_base.predict_proba(X_scaled)[0, 1]

# 3) XGBoost
dmat = xgb.DMatrix(X_scaled, feature_names=FEATURES)
xgb_p = float(xgb_booster.predict(dmat)[0])

# ============================================
# SAFE ENSEMBLE (No meta-model)
# ============================================
ensemble_p = float(np.mean([lr_p, rf_p, xgb_p]))

st.subheader("📊 Predicted 10-Year CVD Risk")
st.metric("Ensemble Risk", f"{ensemble_p*100:.2f}%")

# ============================================
# INTERPRETATION
# ============================================
if ensemble_p < 0.075:
    risk_cat = "🟢 LOW RISK (<7.5%)"
elif ensemble_p < 0.20:
    risk_cat = "🟡 INTERMEDIATE RISK (7.5–20%)"
else:
    risk_cat = "🔴 HIGH RISK (>20%)"

st.subheader("Risk Category")
st.write(f"**{risk_cat}**")

# ============================================
# SHAP EXPLANATION
# ============================================
st.subheader("Feature Importance (SHAP)")

# Use Random Forest for SHAP local explanations
explainer = shap.TreeExplainer(rf_base)
shap_vals = explainer.shap_values(X_imp)[1]  # class 1

# Bar chart
fig, ax = plt.subplots()
shap.summary_plot(shap_vals, X_imp, plot_type="bar", show=False)
st.pyplot(fig)

# ============================================
# FOOTER
# ============================================
st.info("This model uses an ensemble of LR, RF, and XGB base models. "
        "The ensemble method is designed for cross-version compatibility "
        "and safe deployment on Streamlit Cloud.")
