"""SQL-aligned feature engineering tests."""
from __future__ import annotations

import sqlite3
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
    require_finite_number,
    sqlite_round,
)
from model.preprocess import derive_employment_fields  # noqa: E402

FINANCIAL_FIELDS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)


def _sqlite_select_round(value: float, digits: int) -> float:
    conn = sqlite3.connect(":memory:")
    try:
        row = conn.execute("SELECT round(?, ?)", (float(value), int(digits))).fetchone()
        return float(row[0])
    finally:
        conn.close()


def test_financial_features_match_sql_rounding():
    fin = derive_financial_features(60000, 180000, 12000, 170000)
    assert fin["debt_to_income"] == sqlite_round(180000 / 60000, 4) == 3.0
    assert fin["annuity_to_income"] == sqlite_round(12000 / 60000, 4) == 0.2
    assert fin["loan_term_implied"] == sqlite_round(180000 / 12000, 1) == 15.0
    assert fin["ltv_ratio"] == sqlite_round(170000 / 180000, 4) == 0.9444


@pytest.mark.parametrize(
    "value,digits",
    [
        # Ordinary positives
        (1.23456, 4),
        (15.04, 1),
        (0.94444, 4),
        (3.0, 4),
        # Ordinary negatives
        (-1.23456, 4),
        (-15.04, 1),
        (-0.94444, 4),
        # Exact .5 midpoints (SQLite ROUND half-away-from-zero style)
        (2.5, 0),
        (1.5, 0),
        (-1.5, 0),
        (-2.5, 0),
        # One-decimal midpoints
        (1.25, 1),
        (1.35, 1),
        (-1.25, 1),
        # Four-decimal midpoints
        (0.12345, 4),
        (1.23455, 4),
        (-0.12345, 4),
        # Representative floating-point values
        (2.675, 2),
        (2.675, 3),
        # Immediately above / below a midpoint
        (1.2499999, 1),
        (1.2500001, 1),
        (2.4999999, 0),
        (2.5000001, 0),
        (0.94444999, 4),
        (0.94445001, 4),
    ],
)
def test_sqlite_round_matches_stdlib_sqlite3(value, digits):
    assert sqlite_round(value, digits) == _sqlite_select_round(value, digits)


def test_ext_score_sum_coalesce_behavior():
    assert derive_ext_score_sum(0.5, 0.45, 0.5) == 1.45
    assert derive_ext_score_sum(None, 0.45, 0.5) == 0.95
    assert derive_ext_score_sum(float("nan"), 0.45, 0.5) == 0.95


def test_ext_score_sum_rejects_infinity():
    with pytest.raises(FeatureEngineeringError, match="EXT_SOURCE_1"):
        derive_ext_score_sum(float("inf"), 0.45, 0.5)
    with pytest.raises(FeatureEngineeringError, match="EXT_SOURCE_2"):
        derive_ext_score_sum(0.5, float("-inf"), 0.5)
    with pytest.raises(FeatureEngineeringError, match="EXT_SOURCE_3"):
        derive_ext_score_sum(0.5, 0.45, float("inf"))


def test_flag_helpers():
    assert derive_low_ext_score_2(0.29) == 1
    assert derive_low_ext_score_2(0.30) == 0
    assert derive_low_ext_score_3(0.29) == 1
    assert derive_many_children(3) == 1
    assert derive_many_children(2) == 0
    assert derive_high_inquiry_flag(4) == 1
    assert derive_high_inquiry_flag(3) == 0


@pytest.mark.parametrize(
    "helper,args",
    [
        (derive_low_ext_score_2, ("not-a-number",)),
        (derive_low_ext_score_2, (float("nan"),)),
        (derive_low_ext_score_2, (float("inf"),)),
        (derive_low_ext_score_2, (float("-inf"),)),
        (derive_low_ext_score_3, ("bad",)),
        (derive_low_ext_score_3, (float("inf"),)),
        (derive_many_children, ("kids",)),
        (derive_many_children, (float("nan"),)),
        (derive_many_children, (float("inf"),)),
        (derive_high_inquiry_flag, ("many",)),
        (derive_high_inquiry_flag, (float("-inf"),)),
        (derive_high_inquiry_flag, (float("nan"),)),
    ],
)
def test_flag_helpers_reject_nonnumeric_and_nonfinite(helper, args):
    with pytest.raises(FeatureEngineeringError):
        helper(*args)


def test_invalid_denominators_rejected():
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(0, 180000, 12000, 170000)
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(60000, 0, 12000, 170000)
    with pytest.raises(FeatureEngineeringError):
        derive_financial_features(60000, 180000, -1, 170000)


@pytest.mark.parametrize("field_index", range(4))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "not-a-number"])
def test_derive_financial_features_rejects_invalid_sources(field_index, bad):
    args = [60000.0, 180000.0, 12000.0, 170000.0]
    args[field_index] = bad
    with pytest.raises(FeatureEngineeringError) as exc_info:
        derive_financial_features(*args)
    assert FINANCIAL_FIELDS[field_index] in str(exc_info.value)
    assert "ValueError" not in type(exc_info.value).__name__ or isinstance(
        exc_info.value, FeatureEngineeringError
    )


def test_amt_goods_price_nonnumeric_is_feature_engineering_error_not_value_error():
    with pytest.raises(FeatureEngineeringError, match="AMT_GOODS_PRICE"):
        derive_financial_features(60000, 180000, 12000, "seventeen-thousand")


def test_require_finite_number_reports_field_and_value():
    with pytest.raises(FeatureEngineeringError, match="AMT_CREDIT") as exc_info:
        require_finite_number("AMT_CREDIT", float("inf"), constraint="positive")
    assert "inf" in str(exc_info.value).lower() or "INF" in str(exc_info.value)


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
