#!/bin/bash

echo "=============================================="
echo "🔧 Setting up Streamlit deployment environment"
echo "=============================================="

# --- Ensure correct filenames ---
echo "➡️  Renaming Streamlit app to app.py"
if [ -f "streamlit_framingham_app.py" ]; then
    mv streamlit_framingham_app.py app.py
fi

# --- Create runtime.txt (forces Python 3.10 on Streamlit Cloud) ---
echo "➡️  Creating runtime.txt"
cat <<EOF > runtime.txt
3.10
EOF

# --- Create requirements.txt with ABI-safe versions ---
echo "➡️  Creating requirements.txt"
cat <<EOF > requirements.txt
numpy==1.24.4
pandas
xgboost==1.7.5
scikit-learn==1.2.2
imbalanced-learn==0.10.1
shap==0.41.0
matplotlib
joblib
streamlit
EOF

# --- Ensure model_artifacts folder exists ---
echo "➡️  Checking model_artifacts folder"
if [ ! -d "model_artifacts" ]; then
    echo "❌ ERROR: model_artifacts folder missing"
    echo "Please upload your artifacts before deployment."
else
    echo "✔ model_artifacts folder found"
    echo "Contents:"
    ls -lh model_artifacts
fi

# --- Create .streamlit folder and config ---
echo "➡️  Creating .streamlit/config.toml"
mkdir -p .streamlit
cat <<EOF > .streamlit/config.toml
[server]
headless = true
EOF

# --- Commit helpful debug code into app.py (optional) ---
echo "➡️  Adding debug print inside app.py"

if ! grep -q "WORKING DIRECTORY DEBUG" app.py; then
cat <<'EOF' >> app.py

# --------------------------------------------------
# WORKING DIRECTORY DEBUG (auto-added by bash script)
# --------------------------------------------------
import os
import streamlit as st
st.write("🔍 WORKING DIR:", os.getcwd())
st.write("📁 FILES:", os.listdir())
st.write("📁 model_artifacts exists:", os.path.exists("model_artifacts"))
# --------------------------------------------------

EOF
fi

echo "=============================================="
echo "✅ Setup completed. Now run:"
echo "   git add ."
echo "   git commit -m 'Streamlit deployment fixes'"
echo "   git push"
echo ""
echo "Then redeploy your Streamlit app."
echo "=============================================="
