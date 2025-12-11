# app.py — 12-feature, deployable Streamlit app (safe ensemble)
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CVD Risk (12-feature)", layout="centered")
st.title("Cardiovascular 10-Year Risk — (12-feature model)")

ART = "model_artifacts"

# REQUIRED artifact names (update if your files are named differently)
REQUIRED_FILES = {
    "imputer": "imputer.pkl",
    "scaler": "scaler.pkl",
    "lr": "lr_model.pkl",
    "rf": "rf_base.pkl",
    "xgb": "xgb_booster_full.json",
    # optional threshold file
    "threshold": "threshold.txt"
}

# check artifact presence
missing = [v for v in REQUIRED_FILES.values() if not os.path.exists(os.path.join(ART, v))]
if missing:
    st.error(
        "Missing model artifacts in `model_artifacts/`: "
        f"{missing}. Please upload them (imputer, scaler, lr_model, rf_base, xgb_booster_full.json)."
    )
    st.stop()

# load artifacts
@st.cache_resource
def load_models():
    imputer = joblib.load(os.path.join(ART, REQUIRED_FILES["imputer"]))
    scaler = joblib.load(os.path.join(ART, REQUIRED_FILES["scaler"]))
    lr_model = joblib.load(os.path.join(ART, REQUIRED_FILES["lr"]))
    rf_model = joblib.load(os.path.join(ART, REQUIRED_FILES["rf"]))

    # load xgboost booster
    booster = xgb.Booster()
    booster.load_model(os.path.join(ART, REQUIRED_FILES["xgb"]))

    # optional threshold
    thr_path = os.path.join(ART, REQUIRED_FILES["threshold"])
    threshold = None
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r") as f:
                threshold = float(f.read().strip())
        except Exception:
            threshold = None

    return imputer, scaler, lr_model, rf_model, booster, threshold

try:
    imputer, scaler, lr_model, rf_model, xgb_booster, saved_threshold = load_models()
except Exception as e:
    st.error(f"Failed loading model artifacts: {e}")
    st.stop()

# -----------------------
# Feature list (12 features)
# -----------------------
FEATURES = list(imputer.feature_names_in_)  # use the imputer's feature names as source of truth
# Quick sanity check: ensure it's 12 features (expected)
if len(FEATURES) != 12:
    st.warning(f"Imputer reports {len(FEATURES)} features: {FEATURES}. This app expects 12 features. "
               "Proceeding with the imputer feature list.")
st.write("Model expects features:", FEATURES)

# -----------------------
# Sidebar inputs (build from FEATURES)
# -----------------------
st.sidebar.header("Patient inputs (match training features)")

# helper to create inputs based on feature name — use sensible defaults
def input_widget_for(name):
    # integer binary features
    if name in ("male", "currentSmoker", "BPMeds", "diabetes"):
        return st.sidebar.selectbox(f"{name}", [0, 1], index=0)
    # counts
    if name == "cigsPerDay":
        return st.sidebar.number_input("cigsPerDay", 0, 100, 0)
    # small integer
    if name == "age":
        return st.sidebar.number_input("age", 20, 100, 50)
    # continuous ranges
    if name == "totChol":
        return st.sidebar.number_input("totChol", 100, 500, 200)
    if name == "sysBP":
        return st.sidebar.number_input("sysBP", 80, 250, 120)
    if name == "diaBP":
        return st.sidebar.number_input("diaBP", 40, 160, 80)
    if name == "BMI":
        return st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    if name == "heartRate":
        return st.sidebar.number_input("heartRate", 30, 180, 70)
    if name == "glucose":
        return st.sidebar.number_input("glucose", 40, 400, 90)
    # education or any other small-int features
    if name == "education":
        return st.sidebar.number_input("education (1-4)", 1, 4, 1)
    # fallback
    return st.sidebar.text_input(name, "")

# build inputs in same order as FEATURES
values = {}
for col in FEATURES:
    values[col] = input_widget_for(col)

# create input dataframe
input_df = pd.DataFrame([values], columns=FEATURES)
st.subheader("Input features")
st.write(input_df)

# -----------------------
# Preprocess: impute -> verify order -> scale
# -----------------------
try:
    X_imp_arr = imputer.transform(input_df)  # returns np array
except Exception as e:
    st.error(f"Imputer.transform failed: {e}")
    st.stop()

# Convert to dataframe using imputer.feature_names_in_
X_imp = pd.DataFrame(X_imp_arr, columns=imputer.feature_names_in_)

# Force exact order to FEATURES (defensive)
X_imp = X_imp[FEATURES]

# Scale for LR and RF
try:
    X_scaled = scaler.transform(X_imp)
except Exception as e:
    st.error(f"Scaler.transform failed: {e}")
    st.stop()

# -----------------------
# Predictions
# -----------------------
# Logistic Regression (expects scaled)
try:
    p_lr = float(lr_model.predict_proba(X_scaled)[:, 1][0])
except Exception as e:
    st.error(f"LR predict_proba failed: {e}")
    p_lr = float("nan")

# Random Forest (expects scaled here)
try:
    p_rf = float(rf_model.predict_proba(X_scaled)[:, 1][0])
except Exception as e:
    st.error(f"RF predict_proba failed: {e}")
    p_rf = float("nan")

# XGBoost: use imputed (raw) features or scaled depending on how it was trained.
# We'll try with imputed features; if prediction fails, try scaled features.
p_xgb = None
try:
    dmat = xgb.DMatrix(X_imp.values, feature_names=FEATURES)
    p_xgb = float(xgb_booster.predict(dmat)[0])
except Exception:
    try:
        dmat = xgb.DMatrix(X_scaled, feature_names=FEATURES)
        p_xgb = float(xgb_booster.predict(dmat)[0])
    except Exception as e:
        st.error(f"XGBoost prediction failed with both imputed and scaled inputs: {e}")
        p_xgb = float("nan")

# Ensemble (safe average)
probs = [v for v in (p_lr, p_rf, p_xgb) if (v is not None and not np.isnan(v))]
if len(probs) == 0:
    st.error("All model predictions failed.")
    st.stop()

ensemble_p = float(np.mean(probs))

# optional threshold
if saved_threshold is not None:
    threshold = saved_threshold
else:
    threshold = 0.5

label = int(ensemble_p >= threshold)

st.subheader("Predictions")
tbl = pd.DataFrame({
    "model": ["logistic_regression", "random_forest", "xgboost", "ensemble_avg"],
    "probability": [p_lr, p_rf, p_xgb, ensemble_p]
})
st.table(tbl.style.format({"probability": "{:.4f}"}))

st.metric("Ensemble (average) 10-year risk", f"{ensemble_p*100:.2f}%")
st.write(f"Predicted class (threshold={threshold}): **{label}**")

# Risk category (ACC/AHA style)
if ensemble_p < 0.075:
    cat_txt = "LOW (<7.5%)"
elif ensemble_p < 0.20:
    cat_txt = "INTERMEDIATE (7.5–20%)"
else:
    cat_txt = "HIGH (>20%)"
st.write("Risk category:", cat_txt)

# -----------------------
# SHAP explanations (use RF for speed / stability)
# -----------------------
st.subheader("Explainability (SHAP) — Random Forest (local)")

if st.checkbox("Show SHAP explanation"):
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_out = explainer.shap_values(X_imp)
        # shap_out may be list (for classes) or array
        if isinstance(shap_out, list):
            # class 1 shap
            shap_arr = np.array(shap_out[1])
        else:
            shap_arr = np.array(shap_out)

        # shap_arr shape could be (n_samples, n_features) or (n_classes, n_samples, n_features)
        if shap_arr.ndim == 3:
            # pick first sample, last class if exists
            shap_vec = shap_arr[0, :, -1] if shap_arr.shape[-1] > 1 else shap_arr[0, :, 0]
        elif shap_arr.ndim == 2:
            shap_vec = shap_arr[0]
        else:
            shap_vec = shap_arr.flatten()

        shap_df = pd.DataFrame({"feature": FEATURES, "shap_value": shap_vec})
        shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)

        fig, ax = plt.subplots(figsize=(7, max(3, len(FEATURES)*0.25)))
        ax.barh(shap_df["feature"], shap_df["shap_value"])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value")
        ax.set_title("Local SHAP values (Random Forest)")

        st.pyplot(fig)
        st.dataframe(shap_df)
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# -----------------------
# Debug / artifacts
# -----------------------
with st.expander("Debug & artifacts"):
    st.write("Artifacts folder:", os.path.abspath(ART))
    st.write("Available files:", sorted(os.listdir(ART)))
    st.write("Imputer.feature_names_in_:", list(imputer.feature_names_in_))
    st.write("Feature order used by app:", FEATURES)
    st.write("imputed input (first rows):")
    st.write(X_imp)

st.info("If you still see feature mismatch errors, it means the model artifacts were trained on a different column set/order. "
        "In that case choose to retrain models with the same features or upload matching artifacts.")
