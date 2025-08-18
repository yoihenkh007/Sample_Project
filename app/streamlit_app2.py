# app/streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from joblib import load
import matplotlib.pyplot as plt
import os, sys

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import preprocess_bscan
from features import extract_tabular

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="OCT Disease Classifier (Tabular ML)", layout="wide")
st.title("🧠 OCT Disease Classifier — Tabular ML")

# -----------------------------------------------------------
# LOAD MODEL (bundled with repo under /models)
# -----------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb.joblib")

try:
    bundle = load(MODEL_PATH)
    model, classes, feat_names = bundle["model"], bundle["classes"], bundle["feat_names"]
except Exception as e:
    st.error(f"❌ Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

# -----------------------------------------------------------
# SIDEBAR: Documentation
# -----------------------------------------------------------
st.sidebar.title("ℹ️ About this App")
st.sidebar.markdown(
    """
    This tool classifies **OCT B-scans** into retinal disease categories  
    using a **tabular ML pipeline** trained on extracted features.  

    **Pipeline overview**:
    1. **Preprocessing** – Extract ILM & RPE boundaries from OCT images.  
    2. **Feature extraction** – Compute tabular biomarkers (layer thickness, intensity stats, etc.).  
    3. **Model inference** – Predict class probabilities using an XGBoost model.  

    ---
    **Supported OCT formats:** PNG, JPG, TIFF  
    **Output:** Predicted class + feature table
    """
)

# -----------------------------------------------------------
# MAIN APP: Upload OCT image
# -----------------------------------------------------------
uploaded = st.file_uploader("📤 Upload an OCT B-scan (png/jpg/tiff)", type=["png","jpg","jpeg","tif","tiff"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    with st.spinner("🔄 Preprocessing & predicting..."):
        # Preprocess image
        img_p, y_ilm, y_rpe = preprocess_bscan(img)

        # Extract tabular features
        feats = extract_tabular(img_p, y_ilm, y_rpe)
        X = np.array([feats.get(f, np.nan) for f in feat_names], dtype=float).reshape(1,-1)

        # Prediction
        probs = model.predict_proba(X)[0]
        pred = classes[int(np.argmax(probs))]

    # -----------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------
    st.subheader(f"✅ Prediction: **{pred}**")
    st.bar_chart(pd.DataFrame({"Probability": probs}, index=classes))

    # Overlay plot
    fig = plt.figure(figsize=(10,4))
    plt.imshow(img_p, cmap="gray")
    x = np.arange(img_p.shape[1])
    plt.plot(x, y_ilm, linewidth=1)
    plt.plot(x, y_rpe, linewidth=1)

    # Draw band boundaries
    n_upper, n_lower = 5, 8
    y0, y1 = y_ilm, y_rpe
    for b in range(1, n_upper + n_lower):
        yb = (y0 + (y1 - y0) * (b/(n_upper+n_lower))).astype(int)
        plt.plot(x, yb, linewidth=0.5)

    plt.title("Processed OCT with ILM/RPE + 13 bands")
    plt.axis("off")
    st.pyplot(fig)

    # Show extracted features
    st.markdown("**🔎 Extracted Features (first 30):**")
    df_feats = pd.DataFrame([feats]).iloc[:, :30]   # first 30 features as row
    st.dataframe(df_feats, use_container_width=True)

    # Download as CSV
    dfrow = pd.DataFrame([feats])
    csv = dfrow.to_csv(index=False).encode()
    st.download_button("⬇️ Download feature row (CSV)", data=csv,
                       file_name="oct_features.csv", mime="text/csv")
 
