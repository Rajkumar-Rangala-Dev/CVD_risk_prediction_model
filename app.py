import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="CVD Risk Predictor", layout="centered")
st.title("❤️ 10-Year CHD Risk — Stacked ML Model")

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "model_artifacts")

# ============================
# LOAD ARTIFACTS
# ============================
def load_booster(json_path):
    booster = xgb.Booster()
    booster.load_model(json_path)
    return booster

imputer = joblib.load(os.path.join(ARTIFACTS, "imputer.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS, "scaler.pkl"))
lr_model = joblib.load(os.path.join(ARTIFACTS, "lr_model.pkl"))
rf_base = joblib.load(os.path.join(ARTIFACTS, "rf_base.pkl"))
meta_clf = joblib.load(os.path.join(ARTIFACTS, "meta_clf.pkl"))
booster = load_booster(os.path.join(ARTIFACTS, "xgb_booster_full.json"))

with open(os.path.join(ARTIFACTS, "threshold.txt"), "r") as f:
    threshold = float(f.read().strip())

# ============================
# FEATURE ORDER (from imputer)
# ============================
EXPECTED_COLS = [
    "male","age","education","currentSmoker","cigsPerDay","BPMeds",
    "prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP",
    "BMI","heartRate","glucose","pulse_pressure","chol_per_bmi","age_sysbp"
]

# ============================
# USER INPUTS
# ============================
st.sidebar.header("Patient Information")

male = st.sidebar.selectbox("Male?", [1,0], index=0)
age = st.sidebar.number_input("Age", 20, 100, 50)
education = st.sidebar.selectbox("Education level (1–4)", [1,2,3,4])
currentSmoker = st.sidebar.selectbox("Current smoker?", [0,1])
cigsPerDay = st.sidebar.number_input("Cigarettes / day", 0, 80, 0)
BPMeds = st.sidebar.selectbox("On BP meds?", [0,1])
prevalentStroke = st.sidebar.selectbox("History of stroke?", [0,1])
prevalentHyp = st.sidebar.selectbox("Hypertension?", [0,1])
diabetes = st.sidebar.selectbox("Diabetes?", [0,1])

totChol = st.sidebar.number_input("Total Cholesterol", 100, 450, 200)
sysBP = st.sidebar.number_input("Systolic BP", 90, 250, 120)
diaBP = st.sidebar.number_input("Diastolic BP", 50, 150, 80)
BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
heartRate = st.sidebar.number_input("Heart Rate", 40, 150, 75)
glucose = st.sidebar.number_input("Glucose", 40, 400, 90)

# ============================
# BUILD DATAFRAME
# ============================
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
}])

# ENGINEERED FEATURES
input_df["pulse_pressure"] = input_df["sysBP"] - input_df["diaBP"]
input_df["chol_per_bmi"] = input_df["totChol"] / (input_df["BMI"] + 1e-6)
input_df["age_sysbp"] = input_df["age"] * input_df["sysBP"]

# ORDER COLUMNS
input_df = input_df[EXPECTED_COLS]

st.subheader("Input Data")
st.write(input_df)

# ============================
# PREPROCESS
# ============================
X_imp = pd.DataFrame(imputer.transform(input_df), columns=EXPECTED_COLS)
X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=EXPECTED_COLS)

# XGBoost uses raw imputed features
dmat = xgb.DMatrix(X_imp.values, feature_names=EXPECTED_COLS)

# ============================
# BASE MODEL PREDICTIONS
# ============================
p_lr = lr_model.predict_proba(X_scaled)[0,1]
p_rf = rf_base.predict_proba(X_imp)[0,1]
p_xgb = booster.predict(dmat)[0]

# ============================
# STACKED MODEL PREDICTION
# ============================
meta_input = np.array([[p_lr, p_rf, p_xgb]])
p_final = meta_clf.predict_proba(meta_input)[0,1]

label = int(p_final >= threshold)

# ============================
# DISPLAY RESULTS
# ============================
st.subheader("🔎 Model Predictions")
st.write(pd.DataFrame({
    "Model":["Logistic Regression","Random Forest","XGBoost","Stacked Model"],
    "Probability":[p_lr,p_rf,p_xgb,p_final]
}))

st.metric("Final CHD Risk", f"{p_final*100:.2f}%")
st.write(f"Decision Threshold: **{threshold:.3f}**")
st.write(f"Predicted Class: **{label}**")

# ============================
# SHAP EXPLANATION
# ============================
st.subheader("Explainability — XGBoost SHAP")

if st.checkbox("Show SHAP Explanation"):
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_imp)

    shap_df = pd.DataFrame({
        "feature": EXPECTED_COLS,
        "shap_value": shap_values[0]
    }).sort_values("shap_value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(6,8))
    ax.barh(shap_df["feature"], shap_df["shap_value"])
    ax.invert_yaxis()
    st.pyplot(fig)
    st.dataframe(shap_df)

# ============================
# DEBUG INFO
# ============================
with st.expander("Debug Info"):
    st.write("Expected Columns:", EXPECTED_COLS)
    st.write("Imputed:", X_imp)
    st.write("Scaled:", X_scaled)
    st.write("Meta Input:", meta_input)
