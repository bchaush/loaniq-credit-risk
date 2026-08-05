"""Canonical feature engineering matching training SQL rounding."""
from __future__ import annotations

from typing import Any


class FeatureEngineeringError(ValueError):
    """Invalid inputs for SQL-aligned feature derivation."""


def _require_positive_denominator(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(f"{name} must be numeric") from exc
    if number <= 0:
        raise FeatureEngineeringError(
            f"{name} must be positive for ratio derivation; received {value!r}"
        )
    return number


def derive_financial_features(
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
) -> dict[str, float]:
    """Match sql/feature_engineering.sql ROUND formulas exactly."""
    income_f = _require_positive_denominator("AMT_INCOME_TOTAL", income)
    credit_f = _require_positive_denominator("AMT_CREDIT", credit)
    annuity_f = _require_positive_denominator("AMT_ANNUITY", annuity)
    goods_f = float(goods_price)
    if goods_f < 0:
        raise FeatureEngineeringError(
            f"AMT_GOODS_PRICE must be non-negative; received {goods_price!r}"
        )
    return {
        "debt_to_income": round(credit_f / income_f, 4),
        "annuity_to_income": round(annuity_f / income_f, 4),
        "loan_term_implied": round(credit_f / annuity_f, 1),
        "ltv_ratio": round(goods_f / credit_f, 4),
    }


def derive_ext_score_sum(
    ext_source_1: float | None,
    ext_source_2: float | None,
    ext_source_3: float | None,
) -> float:
    """SQL: ROUND(COALESCE(EXT_SOURCE_1,0)+COALESCE(EXT_SOURCE_2,0)+COALESCE(EXT_SOURCE_3,0), 4)."""

    def _coalesce(value: float | None) -> float:
        if value is None:
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise FeatureEngineeringError(
                f"EXT_SOURCE value must be numeric or missing; received {value!r}"
            ) from exc
        if number != number:  # NaN
            return 0.0
        return number

    return round(
        _coalesce(ext_source_1) + _coalesce(ext_source_2) + _coalesce(ext_source_3),
        4,
    )


def derive_low_ext_score_2(ext_source_2: float) -> int:
    return 1 if float(ext_source_2) < 0.30 else 0


def derive_low_ext_score_3(ext_source_3: float) -> int:
    return 1 if float(ext_source_3) < 0.30 else 0


def derive_many_children(cnt_children: float | int) -> int:
    return 1 if int(cnt_children) > 2 else 0


def derive_high_inquiry_flag(credit_inquiries_year: float | int) -> int:
    return 1 if float(credit_inquiries_year) > 3 else 0


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
