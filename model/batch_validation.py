"""Deterministic batch CSV validation for LoanIQ scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model.domain_validation import (
    BINARY_MODEL_FIELDS,
    NONNEGATIVE_INTEGER_FIELDS,
    FeatureEngineeringError,
    validate_binary_value,
    validate_nonnegative_integer,
    validate_region_rating_client,
)
from model.feature_engineering import (
    derive_ext_score_sum,
    derive_financial_features,
    derive_high_inquiry_flag,
    derive_low_ext_score_2,
    derive_low_ext_score_3,
    derive_many_children,
)
from model.preprocess import derive_employment_fields

REQUIRED_COLS = [
    "debt_to_income",
    "annuity_to_income",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "age_years",
    "ext_score_sum",
]

# Continuous engineered fields compared at SQL rounding precision.
ENGINEERED_SPECS = {
    "debt_to_income": {
        "sources": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "atol": 5e-5,  # tighter than 4-decimal unit
    },
    "annuity_to_income": {
        "sources": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "atol": 5e-5,
    },
    "loan_term_implied": {
        "sources": ("AMT_CREDIT", "AMT_ANNUITY"),
        "atol": 5e-2,  # 1-decimal unit
    },
    "ltv_ratio": {
        "sources": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "atol": 5e-5,
    },
}

NUMERIC_MODEL_HINTS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "debt_to_income",
    "annuity_to_income",
    "loan_term_implied",
    "ltv_ratio",
    "age_years",
    "employed_years",
    "employment_to_age_ratio",
    "is_unemployed",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "ext_score_sum",
    "low_ext_score_2",
    "low_ext_score_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "many_children",
    "REGION_RATING_CLIENT",
    "REG_CITY_NOT_WORK_CITY",
    "FLAG_DOCUMENT_3",
    "credit_inquiries_year",
    "high_inquiry_flag",
}

FINANCIAL_SOURCE_FIELDS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)

# Reserved result / internal columns — rejected case-insensitively after strip.
BATCH_OUTPUT_COLUMNS = frozenset(
    {
        "uncalibrated_model_risk_estimate",
        "uncalibrated_model_risk_estimate_display",
        "risk_score",
        "decision",
        "risk_tier",
    }
)
RESERVED_UPLOAD_COLUMNS = BATCH_OUTPUT_COLUMNS | frozenset({"default_probability"})

MAX_ERRORS = 10


def _normalize_column_name(name: Any) -> str:
    return str(name).strip().lower()


def find_reserved_upload_columns(columns: Any) -> list[str]:
    """Return original column names that conflict with reserved result fields.

    Matching is case-insensitive after stripping surrounding whitespace.
    """
    reserved_normalized = {_normalize_column_name(c) for c in RESERVED_UPLOAD_COLUMNS}
    conflicts: list[str] = []
    for col in columns:
        if _normalize_column_name(col) in reserved_normalized:
            conflicts.append(str(col))
    return conflicts



@dataclass
class BatchValidationResult:
    ok: bool
    frame: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    total_error_count: int = 0


def _csv_row(index: Any) -> int:
    try:
        return int(index) + 2  # header is row 1
    except Exception:
        return -1


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def validate_and_prepare_batch(
    frame: pd.DataFrame,
    expected_features: list[str],
) -> BatchValidationResult:
    """Validate/normalize a batch CSV without mutating the input frame.

    Malformed user-supplied values become structured ``BatchValidationResult``
    errors — never raw ``ValueError`` / ``TypeError`` / ``OverflowError`` /
    ``FeatureEngineeringError``.
    """
    if frame is None or not isinstance(frame, pd.DataFrame):
        return BatchValidationResult(
            ok=False,
            errors=["Upload is not a tabular dataframe."],
            total_error_count=1,
        )

    errors: list[str] = []
    notices: list[str] = []
    total_error_count = 0
    # Cells rejected during numeric parse — do not feed into derivation.
    invalid_cells: set[tuple[Any, str]] = set()
    working = frame.copy(deep=True)

    def add_error(message: str) -> None:
        nonlocal total_error_count
        total_error_count += 1
        if len(errors) < MAX_ERRORS:
            errors.append(message)

    if working.empty:
        return BatchValidationResult(
            ok=False,
            errors=["CSV contains no applicant rows."],
            total_error_count=1,
        )
    if len(working) > 5000:
        return BatchValidationResult(
            ok=False,
            errors=["This demo limits batch files to 5,000 rows per run."],
            total_error_count=1,
        )

    reserved_conflicts = find_reserved_upload_columns(working.columns)
    if reserved_conflicts:
        return BatchValidationResult(
            ok=False,
            errors=[
                "Upload contains reserved scoring-output column(s): "
                + ", ".join(reserved_conflicts)
                + ". Remove these columns before upload "
                "(matching is case-insensitive after trimming whitespace)."
            ],
            total_error_count=1,
        )

    missing_required = [c for c in REQUIRED_COLS if c not in working.columns]
    if missing_required:
        return BatchValidationResult(
            ok=False,
            errors=[
                "Missing required columns: "
                + ", ".join(missing_required)
                + ". Download the sample CSV template for the correct column names."
            ],
            total_error_count=1,
        )

    # Convert known numeric columns when present; reject non-empty invalid text.
    candidate_numeric = [
        c for c in working.columns if c in NUMERIC_MODEL_HINTS or c in expected_features
    ]
    # Exclude pure categoricals even if somehow listed.
    categorical = {
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    }
    candidate_numeric = [c for c in candidate_numeric if c not in categorical]

    # Object dtype avoids pandas string-extension refusing int/float repairs.
    for col in candidate_numeric:
        working[col] = working[col].astype(object)

    for col in candidate_numeric:
        for idx, raw in working[col].items():
            blank = isinstance(raw, str) and raw.strip() == ""
            missing = _is_missing(raw) or blank
            # Financial sources: NaN/None/blank are invalid (never silent-missing).
            if col in FINANCIAL_SOURCE_FIELDS and missing:
                add_error(
                    f"Row {_csv_row(idx)} field {col}: non-finite value {raw!r}."
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if col in BINARY_MODEL_FIELDS and missing:
                add_error(
                    f"Row {_csv_row(idx)} field {col}: must be binary 0 or 1; "
                    f"received {raw!r} (expected domain: {{0, 1}})"
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if col in NONNEGATIVE_INTEGER_FIELDS and missing:
                add_error(
                    f"Row {_csv_row(idx)} field {col}: must be a nonnegative integer; "
                    f"received {raw!r} (expected domain: nonnegative integer)"
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if col == "REGION_RATING_CLIENT" and missing:
                add_error(
                    f"Row {_csv_row(idx)} field {col}: must be one of [1, 2, 3]; "
                    f"received {raw!r}"
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if missing:
                if col in REQUIRED_COLS:
                    add_error(
                        f"Row {_csv_row(idx)} field {col}: required value is missing."
                    )
                    invalid_cells.add((idx, col))
                working.at[idx, col] = np.nan
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError):
                add_error(
                    f"Row {_csv_row(idx)} field {col}: non-numeric value {raw!r}."
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if not np.isfinite(number):
                add_error(
                    f"Row {_csv_row(idx)} field {col}: non-finite value {raw!r}."
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue

            # Strict domain checks — never accept fractional binaries/counts via truncation.
            try:
                if col in BINARY_MODEL_FIELDS:
                    working.at[idx, col] = validate_binary_value(raw, col)
                    continue
                if col in NONNEGATIVE_INTEGER_FIELDS:
                    working.at[idx, col] = validate_nonnegative_integer(raw, col)
                    continue
                if col == "REGION_RATING_CLIENT":
                    working.at[idx, col] = validate_region_rating_client(raw, col)
                    continue
            except FeatureEngineeringError as exc:
                add_error(f"Row {_csv_row(idx)} field {col}: {exc}")
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue

            working.at[idx, col] = number

    # Required fields must be finite after conversion (skip cells already flagged).
    for col in REQUIRED_COLS:
        for idx, value in working[col].items():
            if (idx, col) in invalid_cells:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                add_error(
                    f"Row {_csv_row(idx)} field {col}: required value is invalid."
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))
                continue
            if not np.isfinite(number):
                add_error(
                    f"Row {_csv_row(idx)} field {col}: required value is missing or non-finite."
                )
                working.at[idx, col] = np.nan
                invalid_cells.add((idx, col))

    # Source-to-engineered consistency + flag derivation per row.
    for idx, row in working.iterrows():
        row_no = _csv_row(idx)

        # Financial engineered fields — never pass parse-failed sources into derivation.
        income = row.get("AMT_INCOME_TOTAL")
        credit = row.get("AMT_CREDIT")
        annuity = row.get("AMT_ANNUITY")
        goods = row.get("AMT_GOODS_PRICE")
        financial_parse_failed = any(
            (idx, name) in invalid_cells for name in FINANCIAL_SOURCE_FIELDS
        )
        has_financial_sources = (
            not financial_parse_failed
            and all(not _is_missing(v) for v in (income, credit, annuity, goods))
        )
        if has_financial_sources:
            try:
                canon = derive_financial_features(income, credit, annuity, goods)
            except FeatureEngineeringError as exc:
                add_error(f"Row {row_no}: {exc}")
                canon = None
            if canon is not None:
                for field_name, expected in canon.items():
                    atol = ENGINEERED_SPECS[field_name]["atol"]
                    if field_name not in working.columns or _is_missing(row.get(field_name)):
                        working.at[idx, field_name] = expected
                        continue
                    if (idx, field_name) in invalid_cells:
                        continue
                    supplied = float(row.get(field_name))
                    if abs(supplied - expected) > atol:
                        add_error(
                            f"Row {row_no} field {field_name}: supplied {supplied} "
                            f"does not match canonical {expected}."
                        )

        # ext_score_sum — Infinity must never coalesce to zero; skip if any source failed parse.
        ext_invalid = any(
            (idx, name) in invalid_cells
            for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        )
        ext1 = row.get("EXT_SOURCE_1") if "EXT_SOURCE_1" in working.columns else None
        ext2 = row.get("EXT_SOURCE_2")
        ext3 = row.get("EXT_SOURCE_3")
        if (
            not ext_invalid
            and not _is_missing(ext2)
            and not _is_missing(ext3)
        ):
            try:
                expected_sum = derive_ext_score_sum(
                    None if _is_missing(ext1) else ext1,
                    ext2,
                    ext3,
                )
            except FeatureEngineeringError as exc:
                add_error(f"Row {row_no}: {exc}")
                expected_sum = None
            if expected_sum is not None:
                if "ext_score_sum" not in working.columns or _is_missing(
                    row.get("ext_score_sum")
                ):
                    working.at[idx, "ext_score_sum"] = expected_sum
                elif (idx, "ext_score_sum") not in invalid_cells:
                    supplied = float(row.get("ext_score_sum"))
                    if abs(supplied - expected_sum) > 5e-5:
                        add_error(
                            f"Row {row_no} field ext_score_sum: supplied {supplied} "
                            f"does not match canonical {expected_sum}."
                        )

        # Deterministic flags — reject conflicts; normalize valid values to int 0/1
        try:
            if (
                not _is_missing(ext2)
                and (idx, "EXT_SOURCE_2") not in invalid_cells
            ):
                expected_flag = derive_low_ext_score_2(ext2)
                if "low_ext_score_2" in working.columns and not _is_missing(
                    row.get("low_ext_score_2")
                ):
                    try:
                        supplied_flag = validate_binary_value(
                            row.get("low_ext_score_2"), "low_ext_score_2"
                        )
                    except FeatureEngineeringError as exc:
                        add_error(f"Row {row_no} field low_ext_score_2: {exc}")
                        invalid_cells.add((idx, "low_ext_score_2"))
                    else:
                        if supplied_flag != expected_flag:
                            add_error(
                                f"Row {row_no} field low_ext_score_2: supplied "
                                f"{row.get('low_ext_score_2')} conflicts with "
                                f"EXT_SOURCE_2-derived {expected_flag}."
                            )
                        working.at[idx, "low_ext_score_2"] = expected_flag
                else:
                    working.at[idx, "low_ext_score_2"] = expected_flag

            if (
                not _is_missing(ext3)
                and (idx, "EXT_SOURCE_3") not in invalid_cells
            ):
                expected_flag = derive_low_ext_score_3(ext3)
                if "low_ext_score_3" in working.columns and not _is_missing(
                    row.get("low_ext_score_3")
                ):
                    try:
                        supplied_flag = validate_binary_value(
                            row.get("low_ext_score_3"), "low_ext_score_3"
                        )
                    except FeatureEngineeringError as exc:
                        add_error(f"Row {row_no} field low_ext_score_3: {exc}")
                        invalid_cells.add((idx, "low_ext_score_3"))
                    else:
                        if supplied_flag != expected_flag:
                            add_error(
                                f"Row {row_no} field low_ext_score_3: supplied "
                                f"{row.get('low_ext_score_3')} conflicts with "
                                f"EXT_SOURCE_3-derived {expected_flag}."
                            )
                        working.at[idx, "low_ext_score_3"] = expected_flag
                else:
                    working.at[idx, "low_ext_score_3"] = expected_flag

            if (
                "CNT_CHILDREN" in working.columns
                and not _is_missing(row.get("CNT_CHILDREN"))
                and (idx, "CNT_CHILDREN") not in invalid_cells
            ):
                expected_flag = derive_many_children(row.get("CNT_CHILDREN"))
                if "many_children" in working.columns and not _is_missing(
                    row.get("many_children")
                ):
                    try:
                        supplied_flag = validate_binary_value(
                            row.get("many_children"), "many_children"
                        )
                    except FeatureEngineeringError as exc:
                        add_error(f"Row {row_no} field many_children: {exc}")
                        invalid_cells.add((idx, "many_children"))
                    else:
                        if supplied_flag != expected_flag:
                            add_error(
                                f"Row {row_no} field many_children: supplied "
                                f"{row.get('many_children')} conflicts with "
                                f"CNT_CHILDREN-derived {expected_flag}."
                            )
                        working.at[idx, "many_children"] = expected_flag
                else:
                    working.at[idx, "many_children"] = expected_flag

            if (
                "credit_inquiries_year" in working.columns
                and not _is_missing(row.get("credit_inquiries_year"))
                and (idx, "credit_inquiries_year") not in invalid_cells
            ):
                expected_flag = derive_high_inquiry_flag(
                    row.get("credit_inquiries_year")
                )
                if "high_inquiry_flag" in working.columns and not _is_missing(
                    row.get("high_inquiry_flag")
                ):
                    try:
                        supplied_flag = validate_binary_value(
                            row.get("high_inquiry_flag"), "high_inquiry_flag"
                        )
                    except FeatureEngineeringError as exc:
                        add_error(f"Row {row_no} field high_inquiry_flag: {exc}")
                        invalid_cells.add((idx, "high_inquiry_flag"))
                    else:
                        if supplied_flag != expected_flag:
                            add_error(
                                f"Row {row_no} field high_inquiry_flag: supplied "
                                f"{row.get('high_inquiry_flag')} conflicts with "
                                f"credit_inquiries_year-derived {expected_flag}."
                            )
                        working.at[idx, "high_inquiry_flag"] = expected_flag
                else:
                    working.at[idx, "high_inquiry_flag"] = expected_flag
        except FeatureEngineeringError as exc:
            add_error(f"Row {row_no}: {exc}")

        # Employment fields when source information is enough
        income_type = (
            row.get("NAME_INCOME_TYPE")
            if "NAME_INCOME_TYPE" in working.columns
            else None
        )
        age = row.get("age_years")
        emp = row.get("employed_years") if "employed_years" in working.columns else None
        if (
            income_type is not None
            and not _is_missing(age)
            and (idx, "age_years") not in invalid_cells
        ):
            emp_arg = None if _is_missing(emp) else float(emp)
            ey, ratio, unemp = derive_employment_fields(
                str(income_type) if not _is_missing(income_type) else None,
                emp_arg,
                float(age),
            )
            for field_name, expected in (
                ("employed_years", ey),
                ("employment_to_age_ratio", ratio),
                ("is_unemployed", unemp),
            ):
                if field_name not in working.columns or _is_missing(row.get(field_name)):
                    working.at[idx, field_name] = expected
                    continue
                if (idx, field_name) in invalid_cells:
                    continue
                supplied = row.get(field_name)
                if field_name == "is_unemployed":
                    try:
                        supplied_flag = validate_binary_value(
                            supplied, "is_unemployed"
                        )
                    except FeatureEngineeringError as exc:
                        add_error(f"Row {row_no} field is_unemployed: {exc}")
                        invalid_cells.add((idx, "is_unemployed"))
                        continue
                    if supplied_flag != int(expected):
                        add_error(
                            f"Row {row_no} field is_unemployed: supplied {supplied} "
                            f"conflicts with derived {expected}."
                        )
                    working.at[idx, "is_unemployed"] = int(expected)
                else:
                    if (pd.isna(supplied) and pd.isna(expected)) or (
                        np.isfinite(float(supplied))
                        and np.isfinite(float(expected))
                        and abs(float(supplied) - float(expected)) <= 5e-5
                    ):
                        continue
                    if pd.isna(supplied) != pd.isna(expected) or (
                        np.isfinite(float(supplied))
                        and np.isfinite(float(expected))
                        and abs(float(supplied) - float(expected)) > 5e-5
                    ):
                        add_error(
                            f"Row {row_no} field {field_name}: supplied {supplied} "
                            f"conflicts with derived {expected}."
                        )

    if errors:
        if total_error_count > len(errors):
            notices.append(
                f"Showing the first {len(errors)} of "
                f"{total_error_count} validation errors."
            )
        return BatchValidationResult(
            ok=False,
            errors=errors,
            notices=notices,
            total_error_count=total_error_count,
        )

    missing_optional = [c for c in expected_features if c not in working.columns]
    if missing_optional:
        notices.append(
            f"{len(missing_optional)} columns missing from upload and will use "
            "training medians / unknown-category codes. Results may differ from a "
            "fully populated single-applicant profile for affected rows. "
            f"Missing: {', '.join(missing_optional[:5])}"
            + (" and more." if len(missing_optional) > 5 else ".")
        )

    return BatchValidationResult(
        ok=True,
        frame=working,
        errors=[],
        notices=notices,
        total_error_count=0,
    )
