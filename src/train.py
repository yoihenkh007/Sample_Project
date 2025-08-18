# src/train.py
import os, glob, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from joblib import dump

from preprocess import preprocess_bscan
from features import extract_tabular

DISEASES = [
    "AMD", "DME", "ERM", "Normal", "RAO", "RVO", "VMI"
]

def discover_images(root="data/raw"):
    rows = []
    for cls in DISEASES:
        for p in glob.glob(os.path.join(root, cls, "**", "*.*"), recursive=True):
            if p.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                rows.append((p, cls))
    return rows

def build_dataset(root="data/raw", cache_csv="data/processed/features.csv"):
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    records = []
    items = discover_images(root)
    for path, label in tqdm(items, desc="Extracting"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        img_p, y_ilm, y_rpe = preprocess_bscan(img)
        feats = extract_tabular(img_p, y_ilm, y_rpe)
        feats["path"] = path
        feats["label"] = label
        records.append(feats)
    df = pd.DataFrame(records)
    df.to_csv(cache_csv, index=False)
    return df

def train_xgb(cache_csv="data/processed/features.csv", model_path="models/xgb.joblib", feature_list_path="models/features.json"):
    df = pd.read_csv(cache_csv)
    y = df["label"].values
    X = df.drop(columns=["label","path"])
    feat_names = X.columns.tolist()

    # encode labels
    classes = np.unique(y)
    cls2id = {c:i for i,c in enumerate(classes)}
    y_id = np.array([cls2id[c] for c in y])

    # class weights (inverse frequency)
    counts = np.bincount(y_id, minlength=len(classes))
    weights = {i: float(len(y_id)/(len(classes)*max(c,1))) for i,c in enumerate(counts)}
    sample_weight = np.array([weights[i] for i in y_id])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros((len(y_id), len(classes)))
    for fold, (tr, va) in enumerate(skf.split(X, y_id), 1):
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42+fold
        )
        model.fit(X.iloc[tr], y_id[tr], sample_weight=sample_weight[tr])
        oof_pred[va] = model.predict_proba(X.iloc[va])

        print(f"\n=== Fold {fold} Report ===")
        y_hat = np.argmax(oof_pred[va], axis=1)
        print(classification_report(y_id[va], y_hat, target_names=classes))
        print(confusion_matrix(y_id[va], y_hat))

    # final fit on all data
    final = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=123
    )
    final.fit(X, y_id, sample_weight=sample_weight)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump({"model": final, "classes": classes, "feat_names": feat_names, "cls2id": cls2id}, model_path)
    with open(feature_list_path, "w") as f:
        json.dump(feat_names, f, indent=2)
    print(f"Saved: {model_path}")
    return final, classes, feat_names
