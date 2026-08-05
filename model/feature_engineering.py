"""Canonical feature engineering matching training SQL rounding."""
from __future__ import annotations

import math
import sqlite3
from typing import Any, Literal

from model.domain_validation import (
    FeatureEngineeringError,
    validate_nonnegative_integer,
)

Constraint = Literal["finite", "positive", "nonnegative"]

# Back-compat re-export
__all__ = [
    "FeatureEngineeringError",
    "sqlite_round",
    "require_finite_number",
    "derive_financial_features",
    "derive_ext_score_sum",
    "derive_low_ext_score_2",
    "derive_low_ext_score_3",
    "derive_many_children",
    "derive_high_inquiry_flag",
    "build_batch_result_row",
]


_SQLITE = sqlite3.connect(":memory:", check_same_thread=False)


def sqlite_round(value: float, digits: int) -> float:
    """Round using SQLite ROUND semantics (verified by tests against SELECT round(?,?))."""
    row = _SQLITE.execute("SELECT round(?, ?)", (float(value), int(digits))).fetchone()
    return float(row[0])


def require_finite_number(
    name: str,
    value: Any,
    *,
    constraint: Constraint = "finite",
    allow_missing: bool = False,
) -> float | None:
    """Convert and validate a numeric field; never return NaN/Inf."""
    if value is None:
        if allow_missing:
            return None
        raise FeatureEngineeringError(f"{name} is missing; received {value!r}")
    if isinstance(value, str) and value.strip() == "":
        if allow_missing:
            return None
        raise FeatureEngineeringError(f"{name} is blank; received {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureEngineeringError(
            f"{name} must be numeric; received {value!r}"
        ) from exc

    if math.isnan(number):
        if allow_missing:
            return None
        raise FeatureEngineeringError(f"{name} must be finite; received NaN")
    if math.isinf(number):
        raise FeatureEngineeringError(
            f"{name} must be finite; received {value!r}"
        )

    if constraint == "positive" and number <= 0:
        raise FeatureEngineeringError(
            f"{name} must be positive; received {value!r}"
        )
    if constraint == "nonnegative" and number < 0:
        raise FeatureEngineeringError(
            f"{name} must be non-negative; received {value!r}"
        )
    return number


def derive_financial_features(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
) -> dict[str, float]:
    """Match sql/feature_engineering.sql ROUND formulas exactly."""
    income_f = require_finite_number("AMT_INCOME_TOTAL", income, constraint="positive")
    credit_f = require_finite_number("AMT_CREDIT", credit, constraint="positive")
    annuity_f = require_finite_number("AMT_ANNUITY", annuity, constraint="positive")
    goods_f = require_finite_number(
        "AMT_GOODS_PRICE", goods_price, constraint="nonnegative"
    )
    assert income_f is not None and credit_f is not None
    assert annuity_f is not None and goods_f is not None
    return {
        "debt_to_income": sqlite_round(credit_f / income_f, 4),
        "annuity_to_income": sqlite_round(annuity_f / income_f, 4),
        "loan_term_implied": sqlite_round(credit_f / annuity_f, 1),
        "ltv_ratio": sqlite_round(goods_f / credit_f, 4),
    }


def derive_ext_score_sum(
    ext_source_1: float | None,
    ext_source_2: float | None,
    ext_source_3: float | None,
) -> float:
    """SQL: ROUND(COALESCE(EXT_SOURCE_1,0)+COALESCE(EXT_SOURCE_2,0)+COALESCE(EXT_SOURCE_3,0), 4).

    None/NaN coalesce to 0. Infinity is rejected (never treated as missing).
    """

    def _coalesce(name: str, value: float | None) -> float:
        number = require_finite_number(
            name, value, constraint="finite", allow_missing=True
        )
        return 0.0 if number is None else number

    total = (
        _coalesce("EXT_SOURCE_1", ext_source_1)
        + _coalesce("EXT_SOURCE_2", ext_source_2)
        + _coalesce("EXT_SOURCE_3", ext_source_3)
    )
    return sqlite_round(total, 4)


def derive_low_ext_score_2(ext_source_2: float) -> int:
    value = require_finite_number("EXT_SOURCE_2", ext_source_2, constraint="finite")
    assert value is not None
    return 1 if value < 0.30 else 0


def derive_low_ext_score_3(ext_source_3: float) -> int:
    value = require_finite_number("EXT_SOURCE_3", ext_source_3, constraint="finite")
    assert value is not None
    return 1 if value < 0.30 else 0


def derive_many_children(cnt_children: float | int) -> int:
    """SQL: CASE WHEN CNT_CHILDREN > 2 THEN 1 ELSE 0 END (integer count, no truncation)."""
    value = validate_nonnegative_integer(cnt_children, "CNT_CHILDREN")
    return 1 if value > 2 else 0


def derive_high_inquiry_flag(credit_inquiries_year: float | int) -> int:
    value = validate_nonnegative_integer(
        credit_inquiries_year, "credit_inquiries_year"
    )
    return 1 if value > 3 else 0


def build_batch_result_row(score_result: dict[str, Any]) -> dict[str, Any]:
    """User-facing batch columns; never expose default_probability."""
    from model.explainer import format_uncalibrated_risk_display

    prob = float(score_result["default_probability"])
    decision = score_result["decision"]
    return {
        "uncalibrated_model_risk_estimate": prob,
        "uncalibrated_model_risk_estimate_display": format_uncalibrated_risk_display(
            prob, decision
        ),
        "risk_score": score_result["risk_score"],
        "decision": decision,
        "risk_tier": score_result["risk_tier"],
    }
