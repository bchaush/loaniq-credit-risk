"""Integrity tests for LoanIQ train/inference preprocessing and thresholds."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from preprocess import (  # noqa: E402
    DAYS_EMPLOYED_SENTINEL,
    UNKNOWN_CATEGORY_CODE,
    derive_employment_fields,
    employment_to_age_ratio,
    fit_preprocessing,
    transform_applicant,
    transform_frame,
)


def test_preprocessing_fit_on_train_only_medians_and_cats():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 3.0, np.nan],
            "NAME_INCOME_TYPE": ["Working", "Working", "Pensioner"],
            "NAME_EDUCATION_TYPE": ["A", "A", "B"],
            "NAME_FAMILY_STATUS": ["M", "M", "S"],
            "NAME_HOUSING_TYPE": ["H", "H", "H"],
            "OCCUPATION_TYPE": ["Laborers", "Laborers", "Managers"],
            "ORGANIZATION_TYPE": ["X", "X", "Y"],
        }
    )
    art = fit_preprocessing(train, feature_order=list(train.columns))
    assert art["medians"]["x_num"] == pytest.approx(2.0)
    Xv = transform_frame(
        pd.DataFrame(
            {
                "x_num": [np.nan],
                "NAME_INCOME_TYPE": ["BrandNewType"],
                "NAME_EDUCATION_TYPE": ["A"],
                "NAME_FAMILY_STATUS": ["M"],
                "NAME_HOUSING_TYPE": ["H"],
                "OCCUPATION_TYPE": ["Laborers"],
                "ORGANIZATION_TYPE": ["X"],
            }
        ),
        art,
    )
    income_idx = art["feature_order"].index("NAME_INCOME_TYPE")
    assert Xv[0, income_idx] == UNKNOWN_CATEGORY_CODE
    num_idx = art["feature_order"].index("x_num")
    assert Xv[0, num_idx] == pytest.approx(2.0)


def test_early_stopping_uses_validation_not_test_in_train_script():
    text = (ROOT / "model" / "train.py").read_text(encoding="utf-8")
    assert "eval_set=[(X_val_m, y_val)]" in text
    assert "eval_set=[(X_test" not in text
    assert "X_train, X_val, y_train, y_val" in text
    assert "X_test_m" in text and "iteration_range=(0, n_trees_served)" in text
    assert "best_iteration" in text


def test_sentinel_365243_in_sql_and_build():
    sql = (ROOT / "sql" / "feature_engineering.sql").read_text(encoding="utf-8")
    build = (ROOT / "database" / "build_db.py").read_text(encoding="utf-8")
    assert "365243" in sql
    assert str(DAYS_EMPLOYED_SENTINEL) in build or "365243" in build
    assert "DAYS_EMPLOYED = 365243" in sql or "365243" in sql


def test_tenure_ratio_parity():
    assert employment_to_age_ratio(5.0, 40.0) == pytest.approx(0.125)
    days_emp, days_birth = 5.0 * 365.25, 40.0 * 365.25
    sql_ratio = round((days_emp / 365.25) / (days_birth / 365.25), 4)
    assert employment_to_age_ratio(5.0, 40.0) == sql_ratio


def test_missing_numeric_uses_training_median_not_zero():
    train = pd.DataFrame(
        {
            "debt_to_income": [2.0, 4.0, 6.0],
            "NAME_INCOME_TYPE": ["Working"] * 3,
            "NAME_EDUCATION_TYPE": ["A"] * 3,
            "NAME_FAMILY_STATUS": ["M"] * 3,
            "NAME_HOUSING_TYPE": ["H"] * 3,
            "OCCUPATION_TYPE": ["Laborers"] * 3,
            "ORGANIZATION_TYPE": ["X"] * 3,
        }
    )
    art = fit_preprocessing(train, feature_order=list(train.columns))
    row = {
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "A",
        "NAME_FAMILY_STATUS": "M",
        "NAME_HOUSING_TYPE": "H",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "X",
    }
    X = transform_applicant(row, art)
    idx = art["feature_order"].index("debt_to_income")
    assert X[0, idx] == pytest.approx(4.0)
    assert X[0, idx] != 0.0


def test_unseen_category_maps_to_minus_one():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0],
            "NAME_INCOME_TYPE": ["Working", "Working"],
            "NAME_EDUCATION_TYPE": ["A", "A"],
            "NAME_FAMILY_STATUS": ["M", "M"],
            "NAME_HOUSING_TYPE": ["H", "H"],
            "OCCUPATION_TYPE": ["Laborers", "Laborers"],
            "ORGANIZATION_TYPE": ["X", "X"],
        }
    )
    art = fit_preprocessing(train, feature_order=list(train.columns))
    X = transform_applicant(
        {
            "x_num": 1.0,
            "NAME_INCOME_TYPE": "TotallyUnseen",
            "NAME_EDUCATION_TYPE": "A",
            "NAME_FAMILY_STATUS": "M",
            "NAME_HOUSING_TYPE": "H",
            "OCCUPATION_TYPE": "Laborers",
            "ORGANIZATION_TYPE": "X",
        },
        art,
    )
    idx = art["feature_order"].index("NAME_INCOME_TYPE")
    assert X[0, idx] == -1


def test_single_batch_transform_parity():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0, 3.0],
            "NAME_INCOME_TYPE": ["Working"] * 3,
            "NAME_EDUCATION_TYPE": ["A"] * 3,
            "NAME_FAMILY_STATUS": ["M"] * 3,
            "NAME_HOUSING_TYPE": ["H"] * 3,
            "OCCUPATION_TYPE": ["Laborers"] * 3,
            "ORGANIZATION_TYPE": ["X"] * 3,
        }
    )
    art = fit_preprocessing(train, feature_order=list(train.columns))
    row = {
        "x_num": 1.5,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "A",
        "NAME_FAMILY_STATUS": "M",
        "NAME_HOUSING_TYPE": "H",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "X",
    }
    Xs = transform_applicant(row, art)
    Xb = transform_frame(pd.DataFrame([row]), art)
    np.testing.assert_allclose(Xs, Xb)


def test_threshold_boundaries():
    def decide(prob: float) -> str:
        if prob < 0.15:
            return "APPROVED"
        if prob < 0.35:
            return "REVIEW"
        return "DECLINED"

    assert decide(0.1499) == "APPROVED"
    assert decide(0.15) == "REVIEW"
    assert decide(0.3499) == "REVIEW"
    assert decide(0.35) == "DECLINED"


def test_app_py_syntax():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(src)


def test_widget_count_remains_22():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    pattern = re.compile(r"st\.(number_input|selectbox|checkbox)\(")
    assert len(pattern.findall(src)) == 22


def test_no_fillna_zero_in_encode_path():
    expl = (ROOT / "model" / "explainer.py").read_text(encoding="utf-8")
    assert "fillna(0)" not in expl
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "df_batch[col] = 0" not in app


def test_employment_parity_employed_unemployed_pensioner():
    ey, ratio, flag = derive_employment_fields("Working", 5.0, 40.0)
    assert flag == 0
    assert ey == pytest.approx(5.0)
    assert ratio == pytest.approx(0.125)

    ey, ratio, flag = derive_employment_fields("Unemployed", 5.0, 40.0)
    assert flag == 1
    assert np.isnan(ey) and np.isnan(ratio)

    ey, ratio, flag = derive_employment_fields("Pensioner", 8.0, 65.0)
    assert flag == 1
    assert np.isnan(ey) and np.isnan(ratio)

    ey, ratio, flag = derive_employment_fields("Working", float("nan"), 40.0)
    assert flag == 1


def test_inference_uses_best_iteration_not_full_rounds():
    import json
    import joblib
    from model import explainer as expl

    meta = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    assert "best_iteration" in meta
    assert meta["n_trees_served"] == meta["best_iteration"] + 1
    assert expl.N_TREES_SERVED == meta["n_trees_served"]
    assert expl.BEST_ITERATION == meta["best_iteration"]

    src = (ROOT / "model" / "explainer.py").read_text(encoding="utf-8")
    assert "iteration_range=(0, N_TREES_SERVED)" in src

    model = joblib.load(ROOT / "model" / "loaniq_model.pkl")
    assert meta["n_trees_served"] <= 300
    assert meta["n_trees_served"] == meta["best_iteration"] + 1

    X = np.zeros((1, meta["n_features"]))
    p_best = model.predict_proba(X, iteration_range=(0, meta["n_trees_served"]))[0][1]
    p_helper = expl._predict_proba_best(X)
    assert p_helper == pytest.approx(float(p_best))
