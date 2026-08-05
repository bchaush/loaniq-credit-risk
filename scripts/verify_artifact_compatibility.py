#!/usr/bin/env python3
"""Verify LoanIQ artifact load compatibility and golden scoring parity."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.explainer import (  # noqa: E402
    FEATURE_NAMES,
    N_TREES_SERVED,
    _predict_proba_best,
    encode_applicant,
    score_applicant,
)
from model.preprocess import derive_employment_fields  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_live_applicant() -> dict:
    amt_income, amt_annuity, amt_credit, amt_goods = 60000, 12000, 180000, 170000
    dti = round(amt_credit / max(amt_income, 1), 2)
    a2i = round(amt_annuity / max(amt_income, 1), 3)
    ltv = round(amt_goods / max(amt_credit, 1), 3)
    loan_term = round(amt_credit / max(amt_annuity, 1), 0)
    age = 35
    ey, ratio, unemp = derive_employment_fields("Working", 5.0, float(age))
    ext1, ext2, ext3 = 0.50, 0.45, 0.50
    return {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods,
        "debt_to_income": dti,
        "annuity_to_income": a2i,
        "loan_term_implied": loan_term,
        "ltv_ratio": ltv,
        "age_years": float(age),
        "employed_years": ey,
        "employment_to_age_ratio": ratio,
        "is_unemployed": unemp,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "ext_score_sum": ext1 + ext2 + ext3,
        "low_ext_score_2": int(ext2 < 0.3),
        "low_ext_score_3": int(ext3 < 0.3),
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 1,
        "many_children": 0,
        "REGION_RATING_CLIENT": 1,
        "REG_CITY_NOT_WORK_CITY": 0,
        "FLAG_DOCUMENT_3": 1,
        "credit_inquiries_year": 1,
        "high_inquiry_flag": 0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
    }


def main() -> int:
    print("=== LoanIQ artifact compatibility ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"NumPy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")
    print(f"joblib:      {joblib.__version__}")
    print(f"scikit-learn:{sklearn.__version__}")
    print(f"XGBoost:     {xgb.__version__}")

    pre_path = ROOT / "model" / "preprocessing.pkl"
    model_path = ROOT / "model" / "loaniq_model.pkl"
    booster_path = ROOT / "model" / "loaniq_booster.json"
    meta_path = ROOT / "model" / "metadata.json"

    for path in (pre_path, model_path, meta_path, booster_path):
        if not path.exists():
            print(f"FAIL missing artifact: {path}")
            return 1
        print(f"SHA-256 {path.name}: {sha256(path)}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(
        f"features={meta['n_features']} best_iteration={meta.get('best_iteration')} "
        f"n_trees_served={meta.get('n_trees_served')}"
    )
    assert meta["n_features"] == len(FEATURE_NAMES) == 34
    assert int(meta["n_trees_served"]) == N_TREES_SERVED

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joblib.load(pre_path)
        sklearn_model = joblib.load(model_path)
        for item in caught:
            msg = str(item.message)
            print(f"WARNING {item.category.__name__}: {msg}")
            if issubclass(item.category, InconsistentVersionWarning):
                print("FAIL InconsistentVersionWarning is treated as an error")
                return 1
            if "XGBoost" in msg and ("serializ" in msg.lower() or "compat" in msg.lower()):
                print("FAIL XGBoost pickle compatibility warning detected")
                return 1

    # Native booster must match sklearn pickle predictions on golden set.
    native = xgb.Booster()
    native.load_model(str(booster_path))
    applicant = default_live_applicant()
    X = encode_applicant(applicant)
    p_sklearn = float(
        sklearn_model.predict_proba(X, iteration_range=(0, N_TREES_SERVED))[0][1]
    )
    p_native = float(
        native.predict(
            xgb.DMatrix(X, feature_names=FEATURE_NAMES),
            iteration_range=(0, N_TREES_SERVED),
        )[0]
    )
    p_runtime = _predict_proba_best(X)
    print(f"golden pkl={p_sklearn:.10f} json={p_native:.10f} runtime={p_runtime:.10f}")
    if abs(p_sklearn - p_native) > 0.0 or abs(p_runtime - p_native) > 0.0:
        # Allow tiny float noise only if documented; currently require exact match.
        if abs(p_sklearn - p_native) > 1e-12 or abs(p_runtime - p_native) > 1e-12:
            print("FAIL golden prediction parity exceeded tolerance")
            return 1

    scored = score_applicant(applicant)
    prob = scored["default_probability"]
    if not (0.0 <= prob <= 1.0) or not np.isfinite(prob):
        print("FAIL probability not finite in [0, 1]")
        return 1
    print(
        f"default applicant: decision={scored['decision']} "
        f"score={scored['risk_score']} estimate={prob:.4%}"
    )
    if scored["decision"] != "DECLINED" or scored["risk_score"] != 639:
        print("FAIL unexpected default-applicant score/decision")
        return 1
    if abs(prob - 0.3611) > 5e-4:
        print("FAIL unexpected default-applicant uncalibrated estimate")
        return 1

    print("OK artifact compatibility verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
