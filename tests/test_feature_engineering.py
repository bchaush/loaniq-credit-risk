"""SQL-aligned feature engineering tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.explainer import score_applicant  # noqa: E402
from model.feature_engineering import (  # noqa: E402
    FeatureEngineeringError,
    derive_ext_score_sum,
    derive_financial_features,
    derive_high_inquiry_flag,
    derive_low_ext_score_2,
    derive_low_ext_score_3,
    derive_many_children,
)
from model.preprocess import derive_employment_fields  # noqa: E402


def test_financial_features_match_sql_rounding():
    fin = derive_financial_features(60000, 180000, 12000, 170000)
    assert fin["debt_to_income"] == round(180000 / 60000, 4) == 3.0
    assert fin["annuity_to_income"] == round(12000 / 60000, 4) == 0.2
    assert fin["loan_term_implied"] == round(180000 / 12000, 1) == 15.0
    assert fin["ltv_ratio"] == round(170000 / 180000, 4) == 0.9444


def test_ext_score_sum_coalesce_behavior():
    assert derive_ext_score_sum(0.5, 0.45, 0.5) == 1.45
    assert derive_ext_score_sum(None, 0.45, 0.5) == 0.95
    assert derive_ext_score_sum(float("nan"), 0.45, 0.5) == 0.95


def test_flag_helpers():
    assert derive_low_ext_score_2(0.29) == 1
    assert derive_low_ext_score_2(0.30) == 0
    assert derive_low_ext_score_3(0.29) == 1
    assert derive_many_children(3) == 1
    assert derive_many_children(2) == 0
    assert derive_high_inquiry_flag(4) == 1
    assert derive_high_inquiry_flag(3) == 0


def test_invalid_denominators_rejected():
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(0, 180000, 12000, 170000)
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(60000, 0, 12000, 170000)
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(60000, 180000, -1, 170000)


def test_default_applicant_modeled_features_and_score():
    fin = derive_financial_features(60000, 180000, 12000, 170000)
    ey, ratio, unemp = derive_employment_fields("Working", 5.0, 35.0)
    applicant = {
        "AMT_INCOME_TOTAL": 60000,
        "AMT_CREDIT": 180000,
        "AMT_ANNUITY": 12000,
        "AMT_GOODS_PRICE": 170000,
        **fin,
        "age_years": 35.0,
        "employed_years": ey,
        "employment_to_age_ratio": ratio,
        "is_unemployed": unemp,
        "EXT_SOURCE_1": 0.50,
        "EXT_SOURCE_2": 0.45,
        "EXT_SOURCE_3": 0.50,
        "ext_score_sum": derive_ext_score_sum(0.50, 0.45, 0.50),
        "low_ext_score_2": 0,
        "low_ext_score_3": 0,
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
    assert applicant["debt_to_income"] == 3.0
    assert applicant["annuity_to_income"] == 0.2
    assert applicant["loan_term_implied"] == 15.0
    assert applicant["ltv_ratio"] == 0.9444
    scored = score_applicant(applicant)
    assert scored["decision"] == "DECLINED"
    assert scored["risk_score"] == 639
    assert scored["default_probability"] == pytest.approx(0.3611, abs=5e-4)
