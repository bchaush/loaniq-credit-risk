"""Batch validation and output integrity tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.batch_validation import validate_and_prepare_batch  # noqa: E402
from model.explainer import score_applicant  # noqa: E402
from model.feature_engineering import (  # noqa: E402
    build_batch_result_row,
    derive_ext_score_sum,
    derive_financial_features,
    derive_high_inquiry_flag,
    derive_low_ext_score_2,
    derive_low_ext_score_3,
    derive_many_children,
)
from model.preprocess import derive_employment_fields  # noqa: E402
import json

META = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
FEATURES = META["features"]


def default_sample_row() -> dict:
    fin = derive_financial_features(60000, 180000, 12000, 170000)
    ey, ratio, unemp = derive_employment_fields("Working", 5.0, 35.0)
    return {
        "AMT_INCOME_TOTAL": 60000,
        "AMT_CREDIT": 180000,
        "AMT_ANNUITY": 12000,
        "AMT_GOODS_PRICE": 170000,
        **fin,
        "EXT_SOURCE_1": 0.50,
        "EXT_SOURCE_2": 0.45,
        "EXT_SOURCE_3": 0.50,
        "ext_score_sum": derive_ext_score_sum(0.50, 0.45, 0.50),
        "low_ext_score_2": derive_low_ext_score_2(0.45),
        "low_ext_score_3": derive_low_ext_score_3(0.50),
        "age_years": 35,
        "employed_years": ey,
        "employment_to_age_ratio": ratio,
        "is_unemployed": unemp,
        "CNT_CHILDREN": 0,
        "many_children": derive_many_children(0),
        "credit_inquiries_year": 1,
        "high_inquiry_flag": derive_high_inquiry_flag(1),
        "CNT_FAM_MEMBERS": 2,
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 1,
        "REGION_RATING_CLIENT": 1,
        "REG_CITY_NOT_WORK_CITY": 0,
        "FLAG_DOCUMENT_3": 1,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
    }


def test_valid_default_sample_accepted_and_matches_single():
    original = pd.DataFrame([default_sample_row()])
    snapshot = original.copy(deep=True)
    result = validate_and_prepare_batch(original, FEATURES)
    assert result.ok
    assert result.frame is not None
    pd.testing.assert_frame_equal(original, snapshot)  # no in-place mutation
    batch_score = score_applicant(result.frame.iloc[0].to_dict())
    single_score = score_applicant(default_sample_row())
    for key in ("default_probability", "risk_score", "decision", "risk_tier"):
        assert batch_score[key] == single_score[key]
    assert batch_score["decision"] == "DECLINED"
    assert batch_score["risk_score"] == 639


@pytest.mark.parametrize(
    "field,bad",
    [
        ("ext_score_sum", 9.99),
        ("debt_to_income", 9.99),
        ("annuity_to_income", 0.99),
        ("loan_term_implied", 99.0),
        ("ltv_ratio", 0.1),
    ],
)
def test_mismatched_engineered_fields_rejected(field, bad):
    row = default_sample_row()
    row[field] = bad
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert not result.ok
    assert any(field in err for err in result.errors)
    assert any("Row 2" in err for err in result.errors)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("low_ext_score_2", 1),
        ("low_ext_score_3", 1),
        ("many_children", 1),
        ("high_inquiry_flag", 1),
    ],
)
def test_conflicting_flags_rejected(field, bad):
    row = default_sample_row()
    row[field] = bad
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert not result.ok
    assert any(field in err for err in result.errors)


def test_invalid_optional_numeric_text_rejected():
    row = default_sample_row()
    row["AMT_ANNUITY"] = "not-a-number"
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert not result.ok
    assert any("AMT_ANNUITY" in err and "non-numeric" in err for err in result.errors)


def test_infinite_values_rejected():
    row = default_sample_row()
    row["EXT_SOURCE_1"] = float("inf")
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert not result.ok
    assert any("non-finite" in err for err in result.errors)


def test_missing_optional_fields_accepted():
    row = default_sample_row()
    for optional in ("OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_DOCUMENT_3"):
        row.pop(optional, None)
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert result.ok
    assert result.notices  # missing optional notice


def test_unseen_categorical_and_missing_numeric_still_score():
    row = default_sample_row()
    row["NAME_EDUCATION_TYPE"] = "TotallyUnseenEducation"
    row.pop("EXT_SOURCE_1", None)
    # With EXT_SOURCE_1 absent, SQL COALESCE treats it as 0 in the sum.
    row["ext_score_sum"] = derive_ext_score_sum(None, 0.45, 0.50)
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert result.ok
    scored = score_applicant(result.frame.iloc[0].to_dict())
    assert scored["decision"] in {"APPROVED", "REVIEW", "DECLINED"}
    assert 0.0 <= scored["default_probability"] <= 1.0


def test_batch_output_columns_exclude_internal_probability():
    scored = score_applicant(default_sample_row())
    out = build_batch_result_row(scored)
    assert "default_probability" not in out
    assert "uncalibrated_model_risk_estimate" in out
    assert "uncalibrated_model_risk_estimate_display" in out
    assert out["uncalibrated_model_risk_estimate"] == scored["default_probability"]
    assert isinstance(out["uncalibrated_model_risk_estimate"], float)
    assert out["uncalibrated_model_risk_estimate_display"].endswith("%") or out[
        "uncalibrated_model_risk_estimate_display"
    ].startswith("<")
    assert out["risk_score"] == scored["risk_score"]
    assert out["decision"] == scored["decision"]
    assert out["risk_tier"] == scored["risk_tier"]
