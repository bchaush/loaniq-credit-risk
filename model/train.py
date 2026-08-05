"""Train LoanIQ XGBoost with train/val/test discipline and train-only preprocessing."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from preprocess import (  # noqa: E402
    CAT_COLS,
    fit_preprocessing,
    save_preprocessing,
    transform_frame,
)

DB_PATH = "database/loaniq.db"
os.makedirs("model", exist_ok=True)

# ── 1. Load features ──────────────────────────────────────────────
print("Loading features from SQLite...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM model_features WHERE TARGET IS NOT NULL", conn)
conn.close()
print(f"Loaded {len(df):,} rows")

y = df["TARGET"].astype(int)
df = df.drop(columns=["SK_ID_CURR", "TARGET"])

for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype(object)

feature_names = list(df.columns)
print(f"Features: {len(feature_names)}")

# ── 2. Stratified train / validation / test ───────────────────────
# 60% train / 20% val / 20% test (test held out until final metrics)
X_tv, X_test, y_tv, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
)
print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
print(
    f"Default rate — Train: {y_train.mean():.2%}  "
    f"Val: {y_val.mean():.2%}  Test: {y_test.mean():.2%}"
)

# ── 3. Fit preprocessing on TRAIN only ────────────────────────────
artifact = fit_preprocessing(X_train, feature_order=feature_names)
X_train_m = transform_frame(X_train, artifact)
X_val_m = transform_frame(X_val, artifact)
X_test_m = transform_frame(X_test, artifact)

# ── 4. Class weight from y_train only ─────────────────────────────
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.1f}")

# ── 5. Train — early stopping on VALIDATION only ──────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_m,
    y_train,
    eval_set=[(X_val_m, y_val)],
    verbose=50,
)

best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1))
n_trees_served = best_iteration + 1
print(f"\nbest_iteration={best_iteration}  → serving {n_trees_served} trees")

# ── 6. Final metrics on untouched TEST only (trees through best_iteration) ─
y_prob = model.predict_proba(
    X_test_m, iteration_range=(0, n_trees_served)
)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\n{'=' * 40}")
print(f"ROC-AUC (test):  {roc_auc:.4f}")
print(f"PR-AUC (test):   {pr_auc:.4f}")
print(f"{'=' * 40}")
print("\nClassification Report (test):")
print(classification_report(y_test, y_pred))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_pred))

importance = pd.Series(
    model.feature_importances_, index=feature_names
).sort_values(ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string())

model._loaniq_best_iteration = best_iteration
model._loaniq_n_trees_served = n_trees_served

# ── 7. Persist artifacts ──────────────────────────────────────────
joblib.dump(model, "model/loaniq_model.pkl")
joblib.dump(artifact["encoders"], "model/encoders.pkl")
save_preprocessing(artifact, "model/preprocessing.pkl")

metadata = {
    "roc_auc": round(float(roc_auc), 4),
    "pr_auc": round(float(pr_auc), 4),
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "default_rate": round(float(y.mean()), 4),
    "features": feature_names,
    "n_features": len(feature_names),
    "top_features": importance.head(10).index.tolist(),
    "decision_thresholds": {
        "approved_lt": 0.15,
        "review_lt": 0.35,
        "note": "Manually selected demonstration policy bands (not validation-tuned).",
    },
    "early_stopping_eval_set": "validation",
    "best_iteration": best_iteration,
    "n_trees_served": n_trees_served,
    "preprocessing_version": artifact["preprocessing_version"],
}

with open("model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nOK Model saved  →  model/loaniq_model.pkl")
print("OK Encoders saved  →  model/encoders.pkl")
print("OK Preprocessing saved  →  model/preprocessing.pkl")
print("OK Metadata saved  →  model/metadata.json")
