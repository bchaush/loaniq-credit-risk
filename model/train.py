import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json

DB_PATH = "database/loaniq.db"
os.makedirs("model", exist_ok=True)

# ── 1. Load features ──────────────────────────────────────────────
print("Loading features from SQLite...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM model_features WHERE TARGET IS NOT NULL", conn)
conn.close()
print(f"Loaded {len(df):,} rows")

# ── 2. Separate target ────────────────────────────────────────────
y = df["TARGET"].astype(int)
df = df.drop(columns=["SK_ID_CURR", "TARGET"])

# ── 3. Encode categoricals ────────────────────────────────────────
CAT_COLS = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

encoders = {}
for col in CAT_COLS:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── 4. Fill remaining NaNs ────────────────────────────────────────
df = df.fillna(df.median(numeric_only=True))

feature_names = list(df.columns)
print(f"Features: {len(feature_names)}")

# ── 5. Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Default rate — Train: {y_train.mean():.2%}  Test: {y_test.mean():.2%}")

# ── 6. Train XGBoost ──────────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.1f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,   # handles 8% default rate
    use_label_encoder=False,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── 7. Evaluate ───────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc   = roc_auc_score(y_test, y_prob)
pr_auc    = average_precision_score(y_test, y_prob)

print(f"\n{'='*40}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"PR-AUC:   {pr_auc:.4f}")
print(f"{'='*40}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 8. Feature importance (top 15) ───────────────────────────────
importance = pd.Series(
    model.feature_importances_, index=feature_names
).sort_values(ascending=False)

print("\nTop 15 Features:")
print(importance.head(15).to_string())

# ── 9. Save everything ────────────────────────────────────────────
joblib.dump(model, "model/loaniq_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

metadata = {
    "roc_auc":       round(roc_auc, 4),
    "pr_auc":        round(pr_auc, 4),
    "n_train":       int(len(X_train)),
    "n_test":        int(len(X_test)),
    "default_rate":  round(float(y.mean()), 4),
    "features":      feature_names,
    "n_features":    len(feature_names),
    "top_features":  importance.head(10).index.tolist()
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Model saved  →  model/loaniq_model.pkl")
print("✅ Encoders saved  →  model/encoders.pkl")
print("✅ Metadata saved  →  model/metadata.json")