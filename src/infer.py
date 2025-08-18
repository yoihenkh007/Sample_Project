# src/infer.py
import cv2, numpy as np, json
import pandas as pd
from joblib import load
from preprocess import preprocess_bscan
from features import extract_tabular

def load_model(model_path="models/xgb.joblib"):
    bundle = load(model_path)
    return bundle["model"], bundle["classes"], bundle["feat_names"]

def predict_image(path, model_path="models/xgb.joblib"):
    model, classes, feat_names = load_model(model_path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image.")
    img_p, y_ilm, y_rpe = preprocess_bscan(img)
    feats = extract_tabular(img_p, y_ilm, y_rpe)
    X = np.array([feats.get(f, np.nan) for f in feat_names], dtype=float).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "pred_label": classes[pred_idx],
        "probs": {cls: float(p) for cls, p in zip(classes, probs)},
        "features": feats,
        "image_processed": img_p,
        "y_ilm": y_ilm,
        "y_rpe": y_rpe
    }
