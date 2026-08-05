"""Strict domain validators for batch / feature inputs.

Binary and integer helpers never use int(float(x)) truncation as acceptance.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class FeatureEngineeringError(ValueError):
    """Invalid inputs for SQL-aligned feature derivation or domain checks."""


# Proven from model_features / SQL / metadata feature set (0/1 indicators).
BINARY_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "is_unemployed",
        "low_ext_score_2",
        "low_ext_score_3",
        "many_children",
        "high_inquiry_flag",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "FLAG_DOCUMENT_3",
        "REG_CITY_NOT_WORK_CITY",
    }
)

# Proven distinct values in training model_features: {1, 2, 3}.
REGION_RATING_ALLOWED: frozenset[int] = frozenset({1, 2, 3})

# Count fields with nonnegative-integer domain (training evidence + SQL usage).
NONNEGATIVE_INTEGER_FIELDS: frozenset[str] = frozenset(
    {
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "credit_inquiries_year",
    }
)


def validate_binary_value(value: Any, field_name: str) -> int:
    """Accept only exact binary 0/1 (or bool). Reject fractions, NaN, Inf, text."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1) and float(value) in (0.0, 1.0):
            return int(value)
        raise FeatureEngineeringError(
            f"{field_name} must be binary 0 or 1; received {value!r} "
            f"(expected domain: {{0, 1}})"
        )
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise FeatureEngineeringError(
                f"{field_name} must be binary 0 or 1; received {value!r} "
                f"(expected domain: {{0, 1}})"
            )
        if number == 0.0:
            return 0
        if number == 1.0:
            return 1
        raise FeatureEngineeringError(
            f"{field_name} must be binary 0 or 1; received {value!r} "
            f"(expected domain: {{0, 1}})"
        )
    if isinstance(value, str):
        text = value.strip()
        if text == "0":
            return 0
        if text == "1":
            return 1
        raise FeatureEngineeringError(
            f"{field_name} must be binary 0 or 1; received {value!r} "
            f"(expected domain: {{0, 1}})"
        )
    raise FeatureEngineeringError(
        f"{field_name} must be binary 0 or 1; received {value!r} "
        f"(expected domain: {{0, 1}})"
    )


def validate_nonnegative_integer(value: Any, field_name: str) -> int:
    """Accept exact nonnegative integers only (including 0.0)."""
    if isinstance(value, bool):
        raise FeatureEngineeringError(
            f"{field_name} must be a nonnegative integer; received {value!r} "
            f"(expected domain: nonnegative integer)"
        )
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if number < 0:
            raise FeatureEngineeringError(
                f"{field_name} must be a nonnegative integer; received {value!r} "
                f"(expected domain: nonnegative integer)"
            )
        return number
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise FeatureEngineeringError(
                f"{field_name} must be a nonnegative integer; received {value!r} "
                f"(expected domain: nonnegative integer)"
            )
        if number < 0 or number != int(number):
            raise FeatureEngineeringError(
                f"{field_name} must be a nonnegative integer; received {value!r} "
                f"(expected domain: nonnegative integer)"
            )
        return int(number)
    if isinstance(value, str):
        text = value.strip()
        try:
            as_float = float(text)
        except (TypeError, ValueError) as exc:
            raise FeatureEngineeringError(
                f"{field_name} must be a nonnegative integer; received {value!r} "
                f"(expected domain: nonnegative integer)"
            ) from exc
        if not np.isfinite(as_float) or as_float < 0 or as_float != int(as_float):
            raise FeatureEngineeringError(
                f"{field_name} must be a nonnegative integer; received {value!r} "
                f"(expected domain: nonnegative integer)"
            )
        return int(as_float)
    raise FeatureEngineeringError(
        f"{field_name} must be a nonnegative integer; received {value!r} "
        f"(expected domain: nonnegative integer)"
    )


def validate_positive_integer(value: Any, field_name: str) -> int:
    number = validate_nonnegative_integer(value, field_name)
    if number <= 0:
        raise FeatureEngineeringError(
            f"{field_name} must be a positive integer; received {value!r} "
            f"(expected domain: positive integer)"
        )
    return number


def validate_region_rating_client(
    value: Any, field_name: str = "REGION_RATING_CLIENT"
) -> int:
    """Enforce training-proven domain {1, 2, 3}."""
    number = validate_positive_integer(value, field_name)
    if number not in REGION_RATING_ALLOWED:
        raise FeatureEngineeringError(
            f"{field_name} must be one of {sorted(REGION_RATING_ALLOWED)}; "
            f"received {value!r} (expected domain: {sorted(REGION_RATING_ALLOWED)})"
        )
    return number
