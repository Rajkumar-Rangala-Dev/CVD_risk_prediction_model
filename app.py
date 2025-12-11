# app.py — Final deployment-ready Streamlit app for Framingham models
import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional imports (may be heavy)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

st.set_page_config(page_title="Framingham CVD — LR + XGBoost", layout="wide")
st.title("❤️ 10-Year CHD Risk — Logistic Regression + XGBoost")

# -------------------------
# Paths & expected columns
# -------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE, "model_artifacts")

EXPECTED_COLS = [
    "age",
    "sex",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "diabetes",
    "glucose",
    "heartRate"
]

# -------------------------
# Helper: load artifact safely
# -------------------------
def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Try joblib first (pickled sklearn/xgboost objects)
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        # If xgboost Booster saved as model file, try xgboost.Booster().load_model
        if xgb is not None:
            try:
                b = xgb.Booster()
                b.load_model(path)
                return b
            except Exception:
                pass
        # re-raise original joblib error for clarity
        raise e_joblib

# -------------------------
# Load models & artifacts
# -------------------------
missing = []
for fname in ["lr_model.pkl", "xgb_model.pkl", "scaler.pkl"]:
    if not os.path.exists(os.path.join(ARTIFACT_DIR, fname)):
        missing.append(fname)

if missing:
    st.error(f"Missing artifacts in {ARTIFACT_DIR}: {missing}. Upload them and reload.")
    st.stop()

# Load LR and Scaler
try:
    lr_model = load_artifact(os.path.join(ARTIFACT_DIR, "lr_model.pkl"))
except Exception as e:
    st.error(f"Failed loading lr_model.pkl: {e}")
    st.stop()

try:
    scaler = load_artifact(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
except Exception as e:
    st.error(f"Failed loading scaler.pkl: {e}")
    st.stop()

# Load XGBoost (may return Booster or sklearn XGBClassifier)
try:
    xgb_obj = load_artifact(os.path.join(ARTIFACT_DIR, "xgb_model.pkl"))
except Exception as e:
    st.error(f"Failed loading xgb_model.pkl: {e}")
    st.stop()

# Optional: SHAP background
shap_background = None
shap_path = os.path.join(ARTIFACT_DIR, "shap_background.pkl")
if os.path.exists(shap_path):
    try:
        shap_background = joblib.load(shap_path)
    except Exception:
        try:
            shap_background = joblib.load(shap_path, mmap_mode=None)
        except Exception:
            shap_background = None

# Optional metrics
metrics = None
metrics_path = os.path.join(ARTIFACT_DIR, "model_metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None

# -------------------------
# Sidebar: patient inputs (exact training features)
# -------------------------
st.sidebar.header("Patient inputs — match training features")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0], index=0)  # training used sex (numeric)
totChol = st.sidebar.number_input("Total Cholesterol (mg/dL)", 100, 400, 180)
sysBP = st.sidebar.number_input("Systolic BP", 90, 240, 120)
diaBP = st.sidebar.number_input("Diastolic BP", 50, 160, 80)
BMI = st.sidebar.number_input("BMI (kg/m2)", 10.0, 60.0, 25.0, 0.1)
currentSmoker = st.sidebar.selectbox("Current smoker?", [0, 1], index=0)
cigsPerDay = st.sidebar.number_input("Cigarettes per day", 0, 80, 0)
BPMeds = st.sidebar.selectbox("On BP medication?", [0, 1], index=0)
diabetes = st.sidebar.selectbox("Diabetes?", [0, 1], index=0)
glucose = st.sidebar.number_input("Glucose (mg/dL)", 40, 400, 90)
heartRate = st.sidebar.number_input("Heart rate (bpm)", 30, 150, 70)

# Threshold slider for decision
st.sidebar.header("Decision threshold")
threshold = st.sidebar.slider("Threshold for positive (CHD)", 0.0, 1.0, 0.5, 0.01)

# -------------------------
# Build input dataframe in the same order as training
# -------------------------
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
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

# Reorder & validate
try:
    input_df = input_df[EXPECTED_COLS]
except Exception as e:
    st.error(f"Your app expects columns: {EXPECTED_COLS}. Built input columns were: {list(input_df.columns)}. Error: {e}")
    st.stop()

st.subheader("Input summary")
st.table(input_df.T)

# -------------------------
# Preprocess & predict
# -------------------------
# LR needs scaled inputs (trained on StandardScaler)
try:
    X_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Scaler.transform failed: {e}")
    st.stop()

# LR probability
try:
    p_lr = float(lr_model.predict_proba(X_scaled)[:, 1][0])
except Exception as e:
    st.error(f"LogisticRegression predict_proba failed: {e}")
    st.stop()

# XGBoost: if it's a sklearn wrapper (XGBClassifier), call predict_proba on raw features;
# if it's a native Booster, construct DMatrix and call booster.predict
p_xgb = None
try:
    # sklearn wrapper (XGBClassifier or similar)
    if hasattr(xgb_obj, "predict_proba"):
        p_xgb = float(xgb_obj.predict_proba(input_df)[:, 1][0])
    else:
        # native booster
        if xgb is None:
            raise RuntimeError("xgboost not available in runtime")
        dmat = xgb.DMatrix(input_df.values, feature_names=EXPECTED_COLS)
        # Some boosters return array shape (1,) or (n,), take first
        p_xgb = float(xgb_obj.predict(dmat)[0])
except Exception as e:
    st.error(f"XGBoost prediction failed: {e}")
    st.stop()

# Ensemble simple average (or you can choose another logic)
p_ensemble = float(np.mean([p_lr, p_xgb]))

pred_label = int(p_ensemble >= threshold)

# -------------------------
# Display predictions & model probabilities
# -------------------------
st.subheader("🔎 Model predictions (individual & ensemble)")
pred_df = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost", "Ensemble (avg)"],
    "Probability": [p_lr, p_xgb, p_ensemble]
})
st.table(pred_df.style.format({"Probability": "{:.4f}"}))

st.markdown(f"**Ensemble decision (threshold = {threshold:.2f})**: **{pred_label}** ({'CHD risk' if pred_label==1 else 'No CHD risk'})")
st.metric("Ensemble probability", f"{p_ensemble*100:.2f}%")

# -------------------------
# Show training metrics if provided
# -------------------------
st.subheader("📈 Model performance (from training)")
if metrics:
    perf_df = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "Ensemble"],
        "AUC": [metrics.get("lr_auc"), metrics.get("xgb_auc"), metrics.get("ensemble_auc")],
        "Accuracy": [metrics.get("lr_accuracy"), metrics.get("xgb_accuracy"), metrics.get("ensemble_accuracy")]
    })
    st.table(perf_df)
else:
    st.info("No model_metrics.json found in model_artifacts/ — upload it to display AUC/accuracy from training.")

# -------------------------
# SHAP explanations for XGBoost (single sample)
# -------------------------
st.subheader("🧠 SHAP explainability (XGBoost)")

if shap is None:
    st.warning("SHAP not installed in this environment. Install shap in requirements.txt to enable explanations.")
else:
    if st.checkbox("Show SHAP for XGBoost"):
        try:
            # Build explainer depending on xgb object type
            if hasattr(xgb_obj, "predict_proba") and hasattr(xgb_obj, "get_booster"):
                # sklearn wrapper
                explainer = shap.TreeExplainer(xgb_obj.get_booster())
                bkg = shap_background if shap_background is not None else None
                shap_values = explainer.shap_values(input_df, background_dataset=bkg)
            else:
                # native booster
                explainer = shap.TreeExplainer(xgb_obj)
                bkg = shap_background if shap_background is not None else None
                # shap may return various shapes; handle robustly
                shap_out = explainer.shap_values(input_df if bkg is None else (input_df, bkg))
                # normalize to vector
                if isinstance(shap_out, list):
                    shap_vec = np.array(shap_out[-1])[0]
                else:
                    arr = np.array(shap_out)
                    if arr.ndim == 3:  # (1, n_features, n_classes)
                        shap_vec = arr[0, :, -1]
                    elif arr.ndim == 2:
                        shap_vec = arr[0]
                    else:
                        shap_vec = arr
                shap_df = pd.DataFrame({"feature": EXPECTED_COLS, "shap_value": shap_vec})
                shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(shap_df["feature"], shap_df["shap_value"])
                ax.invert_yaxis()
                ax.set_title("XGBoost SHAP (single sample)")
                st.pyplot(fig)
                st.dataframe(shap_df)
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

# -------------------------
# Debug info & artifact listing
# -------------------------
with st.expander("Debug & artifacts"):
    st.write("Working dir:", BASE)
    st.write("Artifact folder exists:", os.path.exists(ARTIFACT_DIR))
    if os.path.exists(ARTIFACT_DIR):
        st.write(sorted(os.listdir(ARTIFACT_DIR)))
    st.write("Input DF columns:", list(input_df.columns))
    st.write("EXPECTED_COLS:", EXPECTED_COLS)

