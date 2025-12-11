import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="CVD Risk", layout="centered", page_icon="❤️")

st.title("❤️ Cardiovascular 10-Year Risk Prediction")

ART = "model_artifacts"

# ========== Load Artifacts ==========
@st.cache_resource
def load_models():
    imputer = joblib.load(os.path.join(ART, "imputer.pkl"))
    scaler = joblib.load(os.path.join(ART, "scaler.pkl"))
    lr_model = joblib.load(os.path.join(ART, "lr_model.pkl"))
    rf_model = joblib.load(os.path.join(ART, "rf_base.pkl"))

    xgb_booster = xgb.Booster()
    xgb_booster.load_model(os.path.join(ART, "xgb_booster_full.json"))

    return imputer, scaler, lr_model, rf_model, xgb_booster

imputer, scaler, lr_model, rf_model, xgb_booster = load_models()

# TRAINING FEATURES — EXACT ORDER
FEATURES = [
 "male","age","education","currentSmoker","cigsPerDay","BPMeds",
 "prevalentStroke","prevalentHyp","diabetes","totChol","sysBP",
 "diaBP","BMI","heartRate","glucose","pulse_pressure",
 "chol_per_bmi","age_sysbp"
]

# ========== Sidebar Inputs ==========
st.sidebar.header("Patient Inputs")

def num(name, minv, maxv, val): 
    return float(st.sidebar.number_input(name, min_value=minv, max_value=maxv, value=val))

def cat(name): 
    return int(st.sidebar.selectbox(name, [0,1]))

male = cat("Male")
age = num("Age", 20, 100, 50)
education = num("Education (1-4)", 1, 4, 1)
currentSmoker = cat("Current Smoker")
cigsPerDay = num("Cigarettes Per Day", 0, 60, 0)
BPMeds = cat("BP Medication")
prevalentStroke = cat("Prior Stroke")
prevalentHyp = cat("Hypertension")
diabetes = cat("Diabetes")
totChol = num("Total Cholesterol", 100, 500, 200)
sysBP = num("Systolic BP", 80, 250, 120)
diaBP = num("Diastolic BP", 40, 150, 80)
BMI = num("BMI", 10.0, 60.0, 25.0)
heartRate = num("Heart Rate", 40, 150, 75)
glucose = num("Glucose", 40, 400, 90)

# DERIVED FEATURES
pulse_pressure = sysBP - diaBP
chol_per_bmi = totChol / BMI
age_sysbp = age * sysBP

# Build input row
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
 "pulse_pressure": pulse_pressure,
 "chol_per_bmi": chol_per_bmi,
 "age_sysbp": age_sysbp
}])

st.subheader("Model Input")
st.write(input_df)

# ========== Preprocess ==========
try:
    X_imp = pd.DataFrame(imputer.transform(input_df), columns=imputer.feature_names_in_)
    X_imp = X_imp[FEATURES]
    X_scaled = scaler.transform(X_imp)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# ========== Predictions ==========
p_lr = lr_model.predict_proba(X_scaled)[0,1]
p_rf = rf_model.predict_proba(X_scaled)[0,1]

dmat = xgb.DMatrix(X_imp.values, feature_names=FEATURES)
p_xgb = float(xgb_booster.predict(dmat)[0])

ensemble_p = np.mean([p_lr, p_rf, p_xgb])

st.subheader("📊 Predicted 10-Year Risk")
st.metric("Risk Probability", f"{ensemble_p*100:.2f}%")

# Risk category
if ensemble_p < 0.075:
    st.write("🟢 **Low Risk (<7.5%)**")
elif ensemble_p < 0.20:
    st.write("🟡 **Intermediate Risk (7.5–20%)**")
else:
    st.write("🔴 **High Risk (>20%)**")

# ========== SHAP Explanation ==========
st.subheader("Feature Importance (SHAP — Random Forest)")

explainer = shap.TreeExplainer(rf_model)
shap_vals = explainer.shap_values(X_imp)[1]

fig, ax = plt.subplots()
shap.summary_plot(shap_vals, X_imp, plot_type="bar", show=False)
st.pyplot(fig)

