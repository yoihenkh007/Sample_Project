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

from src.preprocess import preprocess_bscan
from src.features import extract_tabular

DISEASES = [
    "AMD", "DME", "ERM", "Normal", "RAO", "RVO", "VMI"
]

def discover_images(root="data/raw"):
    rows = []
    for cls in DISEASES:
        folder = os.path.join(root, cls)
        if not os.path.exists(folder):
            print(f"⚠️ WARNING: Folder missing: {folder}")
            continue
        for p in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
            if p.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                rows.append((p, cls))
    return rows

def build_dataset(root="data/raw", cache_csv="data/processed/features.csv"):
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    records = []
    items = discover_images(root)
    print(f"📂 Found {len(items)} images total")
    if len(items) == 0:
        print("❌ No images found! Please check your dataset path.")
        return pd.DataFrame()

    for i, (path, label) in enumerate(items, 1):
        if i % 50 == 0:
            print(f"✅ Processed {i}/{len(items)} images...")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipped unreadable file: {path}")
            continue
        try:
            img_p, y_ilm, y_rpe = preprocess_bscan(img)
            feats = extract_tabular(img_p, y_ilm, y_rpe)
            feats["path"] = path
            feats["label"] = label
            records.append(feats)
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(cache_csv, index=False)
    print(f"💾 Saved features to {cache_csv} with shape {df.shape}")
    return df

def train_xgb(cache_csv="data/processed/features.csv", model_path="models/xgb.joblib", feature_list_path="models/features.json"):
    df = pd.read_csv(cache_csv)
    print(f"📊 Loaded dataset from {cache_csv} with shape {df.shape}")

    y = df["label"].values
    X = df.drop(columns=["label","path"])
    feat_names = X.columns.tolist()

    classes = np.unique(y)
    cls2id = {c:i for i,c in enumerate(classes)}
    y_id = np.array([cls2id[c] for c in y])

    counts = np.bincount(y_id, minlength=len(classes))
    print(f"📈 Class distribution: {dict(zip(classes, counts))}")

    weights = {i: float(len(y_id)/(len(classes)*max(c,1))) for i,c in enumerate(counts)}
    sample_weight = np.array([weights[i] for i in y_id])

    print("🚀 Starting training with 5-fold cross validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(X, y_id), 1):
        print(f"🔄 Fold {fold}...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42+fold
        )
        model.fit(X.iloc[tr], y_id[tr], sample_weight=sample_weight[tr])
        y_hat = model.predict(X.iloc[va])
        print(f"✅ Fold {fold} done. Sample report:")
        print(classification_report(y_id[va], y_hat, target_names=classes, zero_division=0))

    # Train final model
    print("🧠 Training final model on all data...")
    final = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
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

    print(f"💾 Model saved to {model_path}")
    print("🎉 Training complete!")

if __name__ == "__main__":
    print("🏃 Running train.py ...")
    df = build_dataset()
    if not df.empty:
        train_xgb()
    print("✅ Script finished.")
