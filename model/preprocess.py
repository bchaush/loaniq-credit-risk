"""Train-only preprocessing fit + shared transform for single and batch scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PREPROCESSING_VERSION = "1.0"
PREPROCESSING_PATH = Path(__file__).resolve().parent / "preprocessing.pkl"

CAT_COLS = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

UNKNOWN_CATEGORY_CODE = -1
DAYS_EMPLOYED_SENTINEL = 365243

# Income types that imply no active employment tenure in Home Credit training
NO_ACTIVE_TENURE_INCOME_TYPES = frozenset({"Unemployed", "Pensioner"})


def employment_to_age_ratio(employed_years: float, age_years: float) -> float:
    """Application feature-contract ratio: employed_years / age_years.

    Historical training SQL computed the analogous ratio from source day values
    (DAYS_EMPLOYED / DAYS_BIRTH). The deployed app derives it from the
    user-facing year fields (which may already be rounded). Those definitions
    agree for the default applicant and covered parity fixtures; exact identity
    for every theoretical raw record must not be assumed. Changing this serving
    definition requires retraining and parity evaluation.
    """
    age = float(age_years)
    if age <= 0:
        return float("nan")
    return round(float(employed_years) / age, 4)


def derive_employment_fields(
    income_type: str | None,
    employed_years: float | None,
    age_years: float,
) -> tuple[float, float, int]:
    """Derive tenure features for the current serving feature contract.

    No-active tenure → missing employed_years; is_unemployed from tenure.
    Returns (employed_years_feat, employment_to_age_ratio, is_unemployed).
    is_unemployed is 1 iff transformed tenure is missing — not from income label alone.

    The deployed application preserves the feature contract used by the current
    serving artifacts. Historical training SQL computed employment-to-age from
    source day values, while the application derives it from its supplied year
    fields. The serving definition must not be changed without retraining and
    parity evaluation.
    """
    no_active = (
        income_type in NO_ACTIVE_TENURE_INCOME_TYPES
        or employed_years is None
        or (isinstance(employed_years, float) and np.isnan(employed_years))
    )
    if no_active:
        employed_years_feat = float("nan")
        emp_age_r = float("nan")
    else:
        employed_years_feat = float(employed_years)
        emp_age_r = employment_to_age_ratio(employed_years_feat, age_years)

    is_unemployed = 1 if np.isnan(employed_years_feat) else 0
    return employed_years_feat, emp_age_r, is_unemployed


def fit_preprocessing(
    X_train: pd.DataFrame,
    feature_order: list[str] | None = None,
) -> dict[str, Any]:
    """Fit medians and categorical label maps on training rows only."""
    if feature_order is None:
        feature_order = list(X_train.columns)

    encoders: dict[str, LabelEncoder] = {}
    categorical_mappings: dict[str, dict[str, int]] = {}
    train = X_train.copy()

    for col in CAT_COLS:
        if col not in train.columns:
            continue
        series = train[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        le.fit(series)
        encoders[col] = le
        categorical_mappings[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

    numeric = train.drop(columns=[c for c in CAT_COLS if c in train.columns], errors="ignore")
    medians = numeric.median(numeric_only=True).to_dict()
    for col in feature_order:
        if col in CAT_COLS:
            continue
        if col not in medians or pd.isna(medians[col]):
            medians[col] = 0.0
        else:
            medians[col] = float(medians[col])

    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_order": list(feature_order),
        "cat_cols": list(CAT_COLS),
        "medians": medians,
        "categorical_mappings": categorical_mappings,
        "unknown_category_code": UNKNOWN_CATEGORY_CODE,
        "encoders": encoders,
    }


def _encode_series(series: pd.Series, mapping: dict[str, int], unknown: int) -> pd.Series:
    filled = series.fillna("Unknown").astype(str)
    return filled.map(lambda x: mapping[x] if x in mapping else unknown).astype(int)


def transform_frame(df: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    """Transform applicants to the training feature matrix (shared path)."""
    feature_order: list[str] = artifact["feature_order"]
    medians: dict[str, float] = artifact["medians"]
    mappings: dict[str, dict[str, int]] = artifact["categorical_mappings"]
    unknown = int(artifact.get("unknown_category_code", UNKNOWN_CATEGORY_CODE))
    cat_cols = artifact.get("cat_cols", CAT_COLS)

    out = pd.DataFrame(index=df.index)

    for col in feature_order:
        if col in cat_cols:
            if col in df.columns:
                mapping = mappings.get(col, {})
                out[col] = _encode_series(df[col], mapping, unknown)
            else:
                out[col] = unknown
        else:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
            else:
                vals = pd.Series(np.nan, index=df.index)
            median = medians.get(col, 0.0)
            out[col] = vals.fillna(median)

    return out[feature_order].to_numpy(dtype=float)


def transform_applicant(applicant: dict, artifact: dict[str, Any]) -> np.ndarray:
    return transform_frame(pd.DataFrame([applicant]), artifact)


def save_preprocessing(artifact: dict[str, Any], path: Path | str = PREPROCESSING_PATH) -> None:
    """Persist preprocessing.pkl and an inspection-only preprocessing.json sidecar.

    Runtime and verification load the pickle only. The JSON sidecar is for human
    inspection and must be published with the rest of a serving bundle when
    generated (never left unmanaged beside older primary artifacts).
    """
    path = Path(path)
    joblib.dump(artifact, path)
    sidecar = {
        "preprocessing_version": artifact["preprocessing_version"],
        "feature_order": artifact["feature_order"],
        "cat_cols": artifact["cat_cols"],
        "medians": artifact["medians"],
        "categorical_mappings": artifact["categorical_mappings"],
        "unknown_category_code": artifact["unknown_category_code"],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def load_preprocessing(path: Path | str = PREPROCESSING_PATH) -> dict[str, Any]:
    return joblib.load(path)
