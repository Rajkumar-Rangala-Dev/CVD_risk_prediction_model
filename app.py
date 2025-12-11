import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Clinical 10-Year CHD Risk Calculator",
    layout="centered",
    page_icon="❤️"
)

st.title("❤️ Clinical 10-Year CHD Risk Calculator (Calibrated Model)")
st.write("This tool uses a **calibrated logistic regression model**, trained on the Framingham dataset, "
         "to produce clinically realistic 10-year CHD risk estimates.")

ART = "model_artifacts"

# ============================================
# LOAD ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    calibrated_lr = joblib.load(os.path.join(ART, "lr_calibrated.pkl"))
    scaler = joblib.load(os.path.join(ART, "scaler.pkl"))

    with open(os.path.join(ART, "model_metrics.json")) as f:
        metrics = json.load(f)

    return calibrated_lr, scaler, metrics

cal_model, scaler, metrics = load_artifacts()
FEATURES = metrics["features"]
TEST_METRICS = metrics["test"]

# ============================================
# SIDEBAR INPUTS
# ============================================
st.sidebar.header("Patient Information")

def num(label, min_, max_, default):
    return st.sidebar.number_input(label, min_value=min_, max_value=max_, value=default)

def cat(label):
    return st.sidebar.selectbox(label, [0, 1])

male = cat("Male? (1=yes, 0=no)")
age = num("Age", 20, 90, 50)
totChol = num("Total Cholesterol", 120, 400, 200)
sysBP = num("Systolic BP", 80, 250, 120)
diaBP = num("Diastolic BP", 40, 150, 80)
BMI = num("BMI", 15.0, 60.0, 25.0)
currentSmoker = cat("Current Smoker?")
cigsPerDay = num("Cigarettes Per Day", 0, 70, 0)
BPMeds = cat("On BP Medication?")
diabetes = cat("Diabetes?")
glucose = num("Glucose", 50, 350, 90)
heartRate = num("Heart Rate", 40, 150, 75)

# ============================================
# BUILD INPUT DATAFRAME
# ============================================
input_df = pd.DataFrame([{
    "male": male,
    "age": age,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds,
    "diabetes": diabetes,
    "glucose": glucose,
    "heartRate": heartRate
}])

# Ensure only expected features exist
input_df = input_df[FEATURES]

st.subheader("Patient Data")
st.write(input_df)

# ============================================
# MODEL PREDICTION (Calibrated LR)
# ============================================
X_scaled = scaler.transform(input_df)
p = cal_model.predict_proba(X_scaled)[0,1]

st.subheader("📊 Estimated 10-Year CHD Risk")
st.metric("Calibrated Risk", f"{p*100:.2f}%")

# ============================================
# CLINICAL RISK CATEGORIES (ACC/AHA)
# ============================================
if p < 0.075:
    risk_cat = "🟢 LOW RISK (<7.5%)"
elif p < 0.20:
    risk_cat = "🟡 INTERMEDIATE RISK (7.5–20%)"
else:
    risk_cat = "🔴 HIGH RISK (>20%)"

st.subheader("Risk Category")
st.write(f"**{risk_cat}**")

# ============================================
# MODEL PERFORMANCE (TEST SET)
# ============================================
st.subheader("Model Performance (Held-Out Test Set)")
st.write(TEST_METRICS)

# ============================================
# CALIBRATION CURVE
# ============================================
st.subheader("Calibration Curve")
st.image(os.path.join(ART, "calibration_curve.png"))

# ============================================
# FRAMINGHAM-LIKE POINT SCORE
# ============================================
st.subheader("📌 Framingham-Style Point Score")

# Extract LR coefficients
coef = cal_model.calibrated_classifiers_[0].base_estimator.coef_[0]

def compute_points(row):
    points = 0
    detailed = {}
    for feat, w in zip(FEATURES, coef):
        contrib = row[feat] * w
        pts = int(round(contrib * 10))
        points += pts
        detailed[feat] = pts
    return points, detailed

pts, detailed_pts = compute_points(input_df.iloc[0])

st.write(f"**Total Points: {pts} points**")

with st.expander("Detailed Point Breakdown"):
    st.write(pd.DataFrame.from_dict(detailed_pts, orient="index", columns=["Points"]))

# ============================================
# FOOTER
# ============================================
st.info("This model is calibrated on the Framingham dataset but is **not a substitute for professional medical advice**.")
