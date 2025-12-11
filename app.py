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
# APP CONFIG
# ============================================
st.set_page_config(
    page_title="CVD Risk Prediction",
    layout="centered",
    page_icon="❤️"
)

st.title("❤️ Cardiovascular Disease Risk Prediction")
st.write("This app predicts the **10-year CVD risk** using Logistic Regression, Random Forest, and XGBoost models, "
         "and combines them using a safe ensemble method compatible with Streamlit Cloud.")

ART = "model_artifacts"

# ============================================
# LOAD MODELS + ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    imputer = joblib.load(os.path.join(ART, "imputer.pkl"))
    scaler = joblib.load(os.path.join(ART, "scaler.pkl"))

    # Correct model filenames
    lr_model = joblib.load(os.path.join(ART, "lr_model.pkl"))
    rf_base = joblib.load(os.path.join(ART, "rf_base.pkl"))

    # Load XGBoost booster manually
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(ART, "xgb_booster_full.json"))

    return imputer, scaler, lr_model, rf_base, xgb_booster

imputer, scaler, lr_model, rf_base, xgb_booster = load_artifacts()

# ============================================
# SIDEBAR INPUTS (DEFINE ALL VARIABLES)
# ============================================
st.sidebar.header("Patient Information")

def num(label, min_, max_, default):
    return st.sidebar.number_input(label, min_value=min_, max_value=max_, value=default)

def cat(label):
    return st.sidebar.selectbox(label, [0, 1])

male = cat("Male? (1=yes, 0=no)")
age = num("Age", 20, 90, 50)
education = num("Education Level (1–4)", 1, 4, 1)
currentSmoker = cat("Current Smoker?")
cigsPerDay = num("Cigarettes Per Day", 0, 60, 0)
BPMeds = cat("On BP Medication?")
prevalentStroke = cat("History of Stroke?")
prevalentHyp = cat("Hypertension?")
diabetes = cat("Diabetes?")
totChol = num("Total Cholesterol", 100, 600, 200)
sysBP = num("Systolic BP", 80, 250, 120)
diaBP = num("Diastolic BP", 40, 150, 80)
BMI = num("BMI", 10.0, 60.0, 25.0)
heartRate = num("Heart Rate", 40, 150, 75)
glucose = num("Glucose", 40, 400, 90)

# ============================================
# BUILD INPUT DATAFRAME (18 FEATURES)
# ============================================
input_df = pd.DataFrame([{
    "male": male,
    "age": age,
    "education": education,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds,
    "prevalentStroke": prevalentStroke,
    "prevalentHyp": prevalentHyp,
    "diabetes": diabetes,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose,
    # Derived — computed below
    "pulse_pressure": 0.0,
    "chol_per_bmi": 0.0,
    "age_sysbp": 0.0
}])

# ---- DERIVED FEATURES ----
input_df["pulse_pressure"] = input_df["sysBP"] - input_df["diaBP"]
input_df["chol_per_bmi"] = input_df["totChol"] / input_df["BMI"]
input_df["age_sysbp"] = input_df["age"] * input_df["sysBP"]

# ============================================
# ENFORCE EXACT FEATURE ORDER
# ============================================
FEATURES = [
    "male","age","education","currentSmoker","cigsPerDay","BPMeds",
    "prevalentStroke","prevalentHyp","diabetes","totChol","sysBP",
    "diaBP","BMI","heartRate","glucose","pulse_pressure","chol_per_bmi","age_sysbp"
]

input_df = input_df[FEATURES]

st.subheader("Model Input Features")
st.write(input_df)

# ============================================
# IMPUTE + CORRECT FEATURE ORDER
# ============================================
# Impute using training order
X_imp = pd.DataFrame(
    imputer.transform(input_df[FEATURES]),
    columns=imputer.feature_names_in_
)

# Force exact training order
X_imp = X_imp[FEATURES]

# Scale
X_scaled = scaler.transform(X_imp)

# ============================================
# MODEL PREDICTIONS
# ============================================
# 1) Logistic Regression
lr_p = lr_model.predict_proba(X_scaled)[0, 1]

# 2) Random Forest
rf_p = rf_base.predict_proba(X_scaled)[0, 1]

# 3) XGBoost
dmat = xgb.DMatrix(X_scaled, feature_names=FEATURES)
xgb_p = float(xgb_booster.predict(dmat)[0])

# ============================================
# SAFE ENSEMBLE (average)
# ============================================
ensemble_p = float(np.mean([lr_p, rf_p, xgb_p]))

st.subheader("📊 Predicted 10-Year CVD Risk")
st.metric("Ensemble Risk", f"{ensemble_p*100:.2f}%")

# ============================================
# RISK CATEGORY
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
# SHAP EXPLANATION (Random Forest)
# ============================================
st.subheader("Feature Importance (SHAP)")

explainer = shap.TreeExplainer(rf_base)
shap_vals = explainer.shap_values(X_imp)[1]

fig, ax = plt.subplots()
shap.summary_plot(shap_vals, X_imp, plot_type="bar", show=False)
st.pyplot(fig)

# ============================================
# FOOTER
# ============================================
st.info("This model uses an ensemble of Logistic Regression, Random Forest, and XGBoost. "
        "It is engineered for compatibility with Streamlit Cloud and sklearn 1.8.")
