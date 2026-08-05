#!/usr/bin/env python3
"""Verify LoanIQ artifact load compatibility and golden scoring parity."""
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
    _predict_proba_best,
    encode_applicant,
    score_applicant,
)
from model.preprocess import derive_employment_fields  # noqa: E402

TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decide(prob: float) -> str:
    if prob < 0.15:
        return "APPROVED"
    if prob < 0.35:
        return "REVIEW"
    return "DECLINED"


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


def _with_overrides(base: dict, **overrides: Any) -> dict:
    row = deepcopy(base)
    row.update(overrides)
    return row


def golden_applicants() -> list[dict[str, Any]]:
    """Fixed fixtures with observed expected outcomes encoded explicitly."""
    base = default_live_applicant()

    # 1) Live UI default — Declined / 639 / ~36.11%
    default = {
        "id": "default_declined",
        "applicant": base,
        "expected_decision": "DECLINED",
        "expected_risk_score": 639,
        "expected_prob_approx": 0.3611,
        "prob_abs_tol": 5e-4,
    }

    # 2) Low-risk Approve band (established locally; outcome encoded below)
    approve_income, approve_credit = 100000, 30000
    approve_annuity, approve_goods = 3000, 33000
    approve_ext = 0.70
    approve = {
        "id": "low_risk_approved",
        "applicant": _with_overrides(
            base,
            AMT_INCOME_TOTAL=approve_income,
            AMT_CREDIT=approve_credit,
            AMT_ANNUITY=approve_annuity,
            AMT_GOODS_PRICE=approve_goods,
            debt_to_income=round(approve_credit / approve_income, 2),
            annuity_to_income=round(approve_annuity / approve_income, 3),
            ltv_ratio=round(approve_goods / approve_credit, 3),
            loan_term_implied=round(approve_credit / approve_annuity, 0),
            EXT_SOURCE_1=approve_ext,
            EXT_SOURCE_2=approve_ext,
            EXT_SOURCE_3=approve_ext,
            ext_score_sum=3 * approve_ext,
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

    # 3) Medium-risk Review band
    review_income, review_credit = 80000, 150000
    review_annuity, review_goods = 10000, 140000
    review_ext2 = 0.35
    review = {
        "id": "medium_risk_review",
        "applicant": _with_overrides(
            base,
            AMT_INCOME_TOTAL=review_income,
            AMT_CREDIT=review_credit,
            AMT_ANNUITY=review_annuity,
            AMT_GOODS_PRICE=review_goods,
            debt_to_income=round(review_credit / review_income, 2),
            annuity_to_income=round(review_annuity / review_income, 3),
            ltv_ratio=round(review_goods / review_credit, 3),
            loan_term_implied=round(review_credit / review_annuity, 0),
            EXT_SOURCE_1=0.55,
            EXT_SOURCE_2=review_ext2,
            EXT_SOURCE_3=0.55,
            ext_score_sum=0.55 + review_ext2 + 0.55,
            low_ext_score_2=int(review_ext2 < 0.3),
            low_ext_score_3=0,
        ),
        "expected_decision": "REVIEW",
        "expected_risk_score": 713,
        "expected_prob_approx": 0.2870631515979767,
        "prob_abs_tol": 1e-9,
    }

    # 4) Unseen categorical values → unknown-category mapping (-1)
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
        "expected_decision": None,  # band derived from probability
        "expected_risk_score": None,
        "expected_prob_approx": None,
        "prob_abs_tol": None,
        "require_unknown_category_codes": True,
    }

    # 5) Missing numeric fields → training medians via shared preprocessing
    missing = {
        "id": "missing_numeric_medians",
        "applicant": _with_overrides(
            base,
            # Omit several numeric model fields; shared transform fills train medians.
        ),
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


def _check_parity_case(
    case: dict[str, Any],
    sklearn_model: Any,
    native: xgb.Booster,
) -> None:
    applicant = case["applicant"]
    X = encode_applicant(applicant)
    if case.get("require_unknown_category_codes"):
        # Spot-check that at least one categorical column mapped to -1.
        from model.explainer import PREPROCESSING

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

    for label, prob in (
        ("pkl", p_sklearn),
        ("json", p_native),
        ("runtime", p_runtime),
    ):
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

    if case.get("expected_decision") is not None:
        if scored["decision"] != case["expected_decision"]:
            raise AssertionError(
                f"{case['id']}: expected {case['expected_decision']} got {scored['decision']}"
            )
    if case.get("expected_risk_score") is not None:
        if scored["risk_score"] != case["expected_risk_score"]:
            raise AssertionError(
                f"{case['id']}: expected score {case['expected_risk_score']} "
                f"got {scored['risk_score']}"
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
