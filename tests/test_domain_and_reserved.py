"""Binary / integer domain and reserved-column batch validation tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.batch_validation import (  # noqa: E402
    BATCH_OUTPUT_COLUMNS,
    find_reserved_upload_columns,
    validate_and_prepare_batch,
)
from model.domain_validation import (  # noqa: E402
    BINARY_MODEL_FIELDS,
    FeatureEngineeringError,
    validate_binary_value,
    validate_nonnegative_integer,
)
from model.explainer import score_applicant  # noqa: E402
from model.feature_engineering import (  # noqa: E402
    build_batch_result_row,
    derive_high_inquiry_flag,
    derive_many_children,
)
from tests.test_batch_integrity import default_sample_row  # noqa: E402

FEATURES = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))[
    "features"
]

BINARY_REJECTS = (0.9, 1.9, -1, 2, float("nan"), float("inf"), float("-inf"), "nope")


@pytest.mark.parametrize("field", sorted(BINARY_MODEL_FIELDS))
@pytest.mark.parametrize("ok_value", [0, 1, 0.0, 1.0])
def test_binary_fields_accept_exact_zero_one(field, ok_value):
    assert validate_binary_value(ok_value, field) in (0, 1)


@pytest.mark.parametrize("field", sorted(BINARY_MODEL_FIELDS))
@pytest.mark.parametrize("bad", BINARY_REJECTS)
def test_binary_fields_reject_invalid(field, bad):
    with pytest.raises(FeatureEngineeringError):
        validate_binary_value(bad, field)


@pytest.mark.parametrize("field", sorted(BINARY_MODEL_FIELDS))
@pytest.mark.parametrize("bad", [0.9, 1.9, -1, 2, float("nan"), float("inf"), "x"])
def test_batch_rejects_invalid_binary_fields(field, bad):
    row = default_sample_row()
    if field not in row:
        row[field] = 0
    # Keep derived flags consistent when testing ownership binaries.
    row[field] = bad
    if field in {"low_ext_score_2", "low_ext_score_3", "many_children", "high_inquiry_flag"}:
        # Force a supplied invalid binary independent of derivation path.
        pass
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert result.ok is False
    assert any(field in err for err in result.errors)


def test_valid_derived_flags_normalized_to_int():
    row = default_sample_row()
    row["low_ext_score_2"] = 0.0
    row["low_ext_score_3"] = 0.0
    row["many_children"] = 0.0
    row["high_inquiry_flag"] = 0.0
    row["is_unemployed"] = 0.0
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert result.ok
    out = result.frame.iloc[0]
    for field in (
        "low_ext_score_2",
        "low_ext_score_3",
        "many_children",
        "high_inquiry_flag",
        "is_unemployed",
    ):
        assert out[field] in (0, 1)
        assert int(out[field]) == out[field]


def test_cnt_children_rejects_fractional_and_negative():
    for bad in (2.5, -1, 1.2):
        row = default_sample_row()
        row["CNT_CHILDREN"] = bad
        result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
        assert result.ok is False
        assert any("CNT_CHILDREN" in err for err in result.errors)


def test_many_children_from_validated_counts():
    assert derive_many_children(2) == 0
    assert derive_many_children(3) == 1
    with pytest.raises(FeatureEngineeringError):
        derive_many_children(2.5)
    with pytest.raises(FeatureEngineeringError):
        derive_many_children(-1)


def test_inquiry_count_rejects_fractional_and_negative():
    for bad in (3.7, -2, 1.5):
        with pytest.raises(FeatureEngineeringError):
            validate_nonnegative_integer(bad, "credit_inquiries_year")
        with pytest.raises(FeatureEngineeringError):
            derive_high_inquiry_flag(bad)
        row = default_sample_row()
        row["credit_inquiries_year"] = bad
        result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
        assert result.ok is False


@pytest.mark.parametrize(
    "reserved",
    [
        "decision",
        "risk_score",
        "risk_tier",
        "default_probability",
        "uncalibrated_model_risk_estimate",
        " Decision ",
        "RISK_SCORE",
        "Default_Probability",
    ],
)
def test_reserved_output_columns_rejected(reserved):
    row = default_sample_row()
    row[reserved] = "x"
    result = validate_and_prepare_batch(pd.DataFrame([row]), FEATURES)
    assert result.ok is False
    assert any("reserved" in err.lower() for err in result.errors)
    assert find_reserved_upload_columns([reserved])


def test_normal_sample_upload_still_accepted_and_matches_single():
    original = pd.DataFrame([default_sample_row()])
    result = validate_and_prepare_batch(original, FEATURES)
    assert result.ok
    batch_score = score_applicant(result.frame.iloc[0].to_dict())
    single_score = score_applicant(default_sample_row())
    for key in ("default_probability", "risk_score", "decision", "risk_tier"):
        assert batch_score[key] == single_score[key]


def test_batch_output_columns_exact_and_unique():
    scored = score_applicant(default_sample_row())
    out = build_batch_result_row(scored)
    assert set(out.keys()) == set(BATCH_OUTPUT_COLUMNS)
    assert "default_probability" not in out
    assert len(out.keys()) == len(set(out.keys()))
