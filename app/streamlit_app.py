# app/streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from joblib import load
import matplotlib.pyplot as plt
import tempfile
import os, json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from infer import load_model, predict_image
from preprocess import preprocess_bscan
from features import extract_tabular

st.set_page_config(page_title="OCT Disease Classifier (Tabular ML)", layout="wide")
st.title("🧠 OCT Disease Classifier — Tabular ML")

model_path = st.sidebar.text_input("Model path", "models/xgb.joblib")

uploaded = st.file_uploader("Upload a single OCT B-scan (png/jpg/tiff)", type=["png","jpg","jpeg","tif","tiff"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    with st.spinner("Preprocessing & predicting..."):
        try:
            bundle = load(model_path)
            model, classes, feat_names = bundle["model"], bundle["classes"], bundle["feat_names"]
        except Exception as e:
            st.error(f"Failed to load model at {model_path}: {e}")
            st.stop()

        img_p, y_ilm, y_rpe = preprocess_bscan(img)
        feats = extract_tabular(img_p, y_ilm, y_rpe)
        X = np.array([feats.get(f, np.nan) for f in feat_names], dtype=float).reshape(1,-1)

        probs = model.predict_proba(X)[0]
        pred = classes[int(np.argmax(probs))]

    st.subheader(f"Prediction: **{pred}**")
    st.bar_chart(pd.DataFrame({"prob": probs}, index=classes))

    # overlay plot
    fig = plt.figure(figsize=(10,4))
    plt.imshow(img_p, cmap="gray")
    x = np.arange(img_p.shape[1])
    plt.plot(x, y_ilm, linewidth=1)
    plt.plot(x, y_rpe, linewidth=1)
    # draw band boundaries
    n_upper, n_lower = 5, 8
    y0, y1 = y_ilm, y_rpe
    for b in range(1, n_upper + n_lower):
        yb = (y0 + (y1 - y0) * (b/(n_upper+n_lower))).astype(int)
        plt.plot(x, yb, linewidth=0.5)
    plt.title("Processed image with ILM/RPE and 13 bands")
    plt.axis("off")
    st.pyplot(fig)

    # show top features (by absolute z-score)
    st.markdown("**Extracted features (first 30)**")
    st.write(pd.Series(feats).head(30))

    # download features csv
    dfrow = pd.DataFrame([feats])
    csv = dfrow.to_csv(index=False).encode()
    st.download_button("Download feature row (CSV)", data=csv, file_name="oct_features.csv", mime="text/csv")

st.sidebar.markdown("—")
st.sidebar.caption("Train your model with `python -m src.train` then load it here.")
