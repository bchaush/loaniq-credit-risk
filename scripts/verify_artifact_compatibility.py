#!/usr/bin/env python3
"""Verify LoanIQ artifact load compatibility, manifest hashes, and golden parity."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    PREPROCESSING,
    _predict_proba_best,
    encode_applicant,
    score_applicant,
)
from model.feature_engineering import derive_financial_features  # noqa: E402
from model.preprocess import derive_employment_fields  # noqa: E402

TOL = 1e-12
MANIFEST_PATH = ROOT / "model" / "artifact_manifest.json"
REQUIRED_MANIFEST_FIELDS = (
    "python_version",
    "scikit_learn_version",
    "xgboost_version",
    "numpy_version",
    "pandas_version",
    "joblib_version",
    "preprocessing_schema_version",
    "n_features",
    "best_iteration",
    "n_trees_served",
    "inference_path",
    "pickle_reference",
    "artifacts",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decide(prob: float) -> str:
    if prob < 0.15:
        return "APPROVED"
    if prob < 0.35:
        return "REVIEW"
    return "DECLINED"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("manifest root must be an object")
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in data:
            raise AssertionError(f"manifest missing required field: {key}")
    if not isinstance(data["artifacts"], dict) or not data["artifacts"]:
        raise AssertionError("manifest artifacts must be a non-empty object")
    return data


def verify_manifest_artifacts(manifest: dict[str, Any], root: Path = ROOT) -> None:
    for rel, meta in manifest["artifacts"].items():
        path = root / rel
        if not path.exists():
            raise AssertionError(f"manifest artifact missing: {rel}")
        if "sha256" not in meta or "bytes" not in meta:
            raise AssertionError(f"manifest entry incomplete for {rel}")
        digest = sha256(path)
        size = path.stat().st_size
        if digest != meta["sha256"]:
            raise AssertionError(
                f"SHA-256 mismatch for {rel}: expected {meta['sha256']} got {digest}"
            )
        if int(meta["bytes"]) != size:
            raise AssertionError(
                f"byte-size mismatch for {rel}: expected {meta['bytes']} got {size}"
            )


def verify_manifest_runtime_metadata(manifest: dict[str, Any], meta: dict[str, Any]) -> None:
    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        raise AssertionError(f"Python major.minor must be 3.12; got {py.major}.{py.minor}")
    print(f"Python runtime: {sys.version.split()[0]} (major.minor OK)")

    checks = {
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "n_features": meta["n_features"],
        "best_iteration": meta["best_iteration"],
        "n_trees_served": meta["n_trees_served"],
        "preprocessing_schema_version": str(
            PREPROCESSING.get("preprocessing_version", meta.get("preprocessing_version"))
        ),
    }
    for key, actual in checks.items():
        expected = manifest[key]
        if str(expected) != str(actual):
            raise AssertionError(f"manifest {key}={expected!r} != runtime/meta {actual!r}")


def default_live_applicant() -> dict:
    fin = derive_financial_features(60000, 180000, 12000, 170000)
    age = 35
    ey, ratio, unemp = derive_employment_fields("Working", 5.0, float(age))
    ext1, ext2, ext3 = 0.50, 0.45, 0.50
    return {
        "AMT_INCOME_TOTAL": 60000,
        "AMT_CREDIT": 180000,
        "AMT_ANNUITY": 12000,
        "AMT_GOODS_PRICE": 170000,
        **fin,
        "age_years": float(age),
        "employed_years": ey,
        "employment_to_age_ratio": ratio,
        "is_unemployed": unemp,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "ext_score_sum": round(ext1 + ext2 + ext3, 4),
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


def _with_overrides(base: dict, **overrides: Any) -> dict:
    row = deepcopy(base)
    row.update(overrides)
    return row


def golden_applicants() -> list[dict[str, Any]]:
    base = default_live_applicant()
    default = {
        "id": "default_declined",
        "applicant": base,
        "expected_decision": "DECLINED",
        "expected_risk_score": 639,
        "expected_prob_approx": 0.3611,
        "prob_abs_tol": 5e-4,
    }
    approve = {
        "id": "low_risk_approved",
        "applicant": _with_overrides(
            base,
            AMT_INCOME_TOTAL=100000,
            AMT_CREDIT=30000,
            AMT_ANNUITY=3000,
            AMT_GOODS_PRICE=33000,
            **derive_financial_features(100000, 30000, 3000, 33000),
            EXT_SOURCE_1=0.70,
            EXT_SOURCE_2=0.70,
            EXT_SOURCE_3=0.70,
            ext_score_sum=2.1,
            low_ext_score_2=0,
            low_ext_score_3=0,
            NAME_EDUCATION_TYPE="Higher education",
            FLAG_OWN_CAR=1,
        ),
        "expected_decision": "APPROVED",
        "expected_risk_score": 983,
        "expected_prob_approx": 0.016994252800941467,
        "prob_abs_tol": 1e-9,
    }
    review = {
        "id": "medium_risk_review",
        "applicant": _with_overrides(
            base,
            AMT_INCOME_TOTAL=80000,
            AMT_CREDIT=150000,
            AMT_ANNUITY=10000,
            AMT_GOODS_PRICE=140000,
            **derive_financial_features(80000, 150000, 10000, 140000),
            EXT_SOURCE_1=0.55,
            EXT_SOURCE_2=0.35,
            EXT_SOURCE_3=0.55,
            ext_score_sum=1.45,
            low_ext_score_2=0,
            low_ext_score_3=0,
        ),
        "expected_decision": "REVIEW",
        "expected_risk_score": 713,
        "expected_prob_approx": 0.2870631515979767,
        "prob_abs_tol": 1e-9,
    }
    unseen = {
        "id": "unseen_categoricals",
        "applicant": _with_overrides(
            base,
            NAME_INCOME_TYPE="BrandNewIncomeType",
            NAME_EDUCATION_TYPE="BrandNewEducation",
            NAME_FAMILY_STATUS="BrandNewFamilyStatus",
            NAME_HOUSING_TYPE="BrandNewHousing",
            OCCUPATION_TYPE="BrandNewOccupation",
            ORGANIZATION_TYPE="BrandNewOrganization",
        ),
        "expected_decision": None,
        "expected_risk_score": None,
        "expected_prob_approx": None,
        "prob_abs_tol": None,
        "require_unknown_category_codes": True,
    }
    missing = {
        "id": "missing_numeric_medians",
        "applicant": _with_overrides(base),
        "expected_decision": None,
        "expected_risk_score": None,
        "expected_prob_approx": None,
        "prob_abs_tol": None,
        "drop_numeric_keys": (
            "EXT_SOURCE_1",
            "debt_to_income",
            "annuity_to_income",
            "loan_term_implied",
            "employed_years",
            "employment_to_age_ratio",
        ),
    }
    for key in missing["drop_numeric_keys"]:
        missing["applicant"].pop(key, None)
    return [default, approve, review, unseen, missing]


def _check_parity_case(case: dict[str, Any], sklearn_model: Any, native: xgb.Booster) -> None:
    applicant = case["applicant"]
    X = encode_applicant(applicant)
    if case.get("require_unknown_category_codes"):
        cat_idxs = [
            PREPROCESSING["feature_order"].index(c)
            for c in PREPROCESSING["cat_cols"]
            if c in PREPROCESSING["feature_order"]
        ]
        assert any(int(X[0, i]) == -1 for i in cat_idxs), case["id"]

    p_sklearn = float(
        sklearn_model.predict_proba(X, iteration_range=(0, N_TREES_SERVED))[0][1]
    )
    p_native = float(
        native.predict(
            xgb.DMatrix(X, feature_names=FEATURE_NAMES),
            iteration_range=(0, N_TREES_SERVED),
        )[0]
    )
    p_runtime = float(_predict_proba_best(X))
    for label, prob in (("pkl", p_sklearn), ("json", p_native), ("runtime", p_runtime)):
        if not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
            raise AssertionError(f"{case['id']}: {label} probability out of range: {prob}")
    if abs(p_sklearn - p_native) > TOL or abs(p_runtime - p_native) > TOL:
        raise AssertionError(
            f"{case['id']}: parity exceeded TOL={TOL} "
            f"pkl={p_sklearn} json={p_native} runtime={p_runtime}"
        )

    scored = score_applicant(applicant)
    if abs(scored["default_probability"] - p_runtime) > TOL:
        raise AssertionError(f"{case['id']}: score_applicant probability mismatch")
    if scored["decision"] != _decide(scored["default_probability"]):
        raise AssertionError(f"{case['id']}: decision-band mismatch")
    if case.get("expected_decision") is not None and scored["decision"] != case["expected_decision"]:
        raise AssertionError(
            f"{case['id']}: expected {case['expected_decision']} got {scored['decision']}"
        )
    if case.get("expected_risk_score") is not None and scored["risk_score"] != case["expected_risk_score"]:
        raise AssertionError(
            f"{case['id']}: expected score {case['expected_risk_score']} got {scored['risk_score']}"
        )
    if case.get("expected_prob_approx") is not None:
        if abs(scored["default_probability"] - case["expected_prob_approx"]) > case["prob_abs_tol"]:
            raise AssertionError(
                f"{case['id']}: expected ~{case['expected_prob_approx']} "
                f"got {scored['default_probability']}"
            )
    print(
        f"OK {case['id']}: decision={scored['decision']} "
        f"score={scored['risk_score']} p={scored['default_probability']:.10f} "
        f"parity(pkl/json/runtime)"
    )


def main() -> int:
    print("=== LoanIQ artifact compatibility ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"NumPy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")
    print(f"joblib:      {joblib.__version__}")
    print(f"scikit-learn:{sklearn.__version__}")
    print(f"XGBoost:     {xgb.__version__}")

    try:
        manifest = load_manifest()
        verify_manifest_artifacts(manifest)
        print("OK manifest artifact hashes/sizes")
    except AssertionError as exc:
        print(f"FAIL manifest: {exc}")
        return 1

    meta_path = ROOT / "model" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        verify_manifest_runtime_metadata(manifest, meta)
        print("OK manifest runtime/metadata fields")
    except AssertionError as exc:
        print(f"FAIL manifest metadata: {exc}")
        return 1

    print(
        f"features={meta['n_features']} best_iteration={meta.get('best_iteration')} "
        f"n_trees_served={meta.get('n_trees_served')}"
    )
    assert meta["n_features"] == len(FEATURE_NAMES) == 34
    assert int(meta["n_trees_served"]) == N_TREES_SERVED

    pre_path = ROOT / "model" / "preprocessing.pkl"
    model_path = ROOT / "model" / "loaniq_model.pkl"
    booster_path = ROOT / "model" / "loaniq_booster.json"

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

    native = xgb.Booster()
    native.load_model(str(booster_path))
    cases = golden_applicants()
    assert len(cases) >= 5
    try:
        for case in cases:
            _check_parity_case(case, sklearn_model, native)
    except AssertionError as exc:
        print(f"FAIL {exc}")
        return 1

    print(f"OK artifact compatibility verified ({len(cases)} golden applicants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
