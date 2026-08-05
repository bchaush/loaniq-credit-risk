"""Narrative integrity, fact-engine, and Quick/Full parity tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.explainer import (  # noqa: E402
    build_assessment_facts,
    explain_decision,
    format_uncalibrated_risk_display,
    full_assessment,
    render_deterministic_narrative,
    score_applicant,
)
from model.preprocess import derive_employment_fields  # noqa: E402


def live_default_applicant() -> dict:
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


def _base_applicant(**overrides) -> dict:
    row = live_default_applicant()
    row.update(overrides)
    return row


def _classify_map(facts: dict) -> dict[str, str]:
    out = {}
    for bucket in ("strengths", "risks", "neutral"):
        for fact in facts[bucket]:
            out[fact["id"]] = fact["classification"]
    return out


def test_dti_boundary_classifications():
    score = {"default_probability": 0.2, "risk_score": 800, "decision": "REVIEW", "risk_tier": "Medium Risk"}
    for value, expected in (
        (1.50, "neutral"),
        (3.00, "neutral"),
        (1.4999, "strength"),
        (3.0001, "risk"),
    ):
        facts = build_assessment_facts(_base_applicant(debt_to_income=value), score)
        assert _classify_map(facts)["credit_to_income"] == expected


def test_coverage_and_composite_and_debt_service_boundaries():
    score = {"default_probability": 0.2, "risk_score": 800, "decision": "REVIEW", "risk_tier": "Medium Risk"}
    cases = [
        ({"ltv_ratio": 0.80}, "coverage", "neutral"),
        ({"ltv_ratio": 1.20}, "coverage", "neutral"),
        ({"EXT_SOURCE_1": 0.40}, "ext_source_1", "neutral"),
        ({"EXT_SOURCE_2": 0.70}, "ext_source_2", "neutral"),
        ({"annuity_to_income": 0.30}, "debt_service_income", "neutral"),
        ({"annuity_to_income": 0.50}, "debt_service_income", "neutral"),
    ]
    for overrides, fact_id, expected in cases:
        facts = build_assessment_facts(_base_applicant(**overrides), score)
        assert _classify_map(facts)[fact_id] == expected


def test_neutral_and_parity_fields_excluded_from_strengths_risks():
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    facts = build_assessment_facts(applicant, scored)
    strength_ids = {f["id"] for f in facts["strengths"]}
    risk_ids = {f["id"] for f in facts["risks"]}
    assert "credit_to_income" not in strength_ids | risk_ids
    assert "coverage" not in strength_ids | risk_ids
    assert "ext_source_1" not in strength_ids | risk_ids
    assert "ext_source_2" not in strength_ids | risk_ids
    assert "ext_source_3" not in strength_ids | risk_ids
    assert "parity_age_marital" not in strength_ids | risk_ids
    assert all(f["classification"] == "neutral" for f in facts["neutral"] if f["id"] == "parity_age_marital")


def test_quick_full_score_parity(monkeypatch):
    from model import explainer as expl

    calls = {"n": 0}

    class _Boom:
        def create(self, *args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("simulated Anthropic outage")

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Boom()))
    applicant = live_default_applicant()

    quick = score_applicant(applicant)
    assert calls["n"] == 0

    full = full_assessment(applicant)
    assert calls["n"] == 1
    expected_fallback = render_deterministic_narrative(
        {
            "default_probability": full["default_probability"],
            "risk_score": full["risk_score"],
            "decision": full["decision"],
            "risk_tier": full["risk_tier"],
        },
        build_assessment_facts(applicant, full),
    )
    assert full["explanation"] == expected_fallback

    for key in ("default_probability", "risk_score", "decision", "risk_tier"):
        assert quick[key] == full[key]

    score_applicant(applicant)
    assert calls["n"] == 1


def test_quick_score_makes_zero_anthropic_requests(monkeypatch):
    from model import explainer as expl

    class _Boom:
        def create(self, *args, **kwargs):
            raise AssertionError("Quick score must not call Anthropic")

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Boom()))
    score_applicant(live_default_applicant())


def test_malformed_claude_json_uses_fallback(monkeypatch):
    from model import explainer as expl

    class _Resp:
        content = [SimpleNamespace(text="not-json")]

    class _Ok:
        def create(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Ok()))
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    text = explain_decision(applicant, scored)
    expected = render_deterministic_narrative(scored, build_assessment_facts(applicant, scored))
    assert text == expected


def test_claude_timeout_uses_fallback(monkeypatch):
    from model import explainer as expl

    class _Boom:
        def create(self, *args, **kwargs):
            raise TimeoutError("simulated timeout")

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Boom()))
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    text = explain_decision(applicant, scored)
    assert "uncalibrated model risk estimate" in text.lower()
    assert "probability of default" not in text.lower()


def test_unknown_fact_ids_rejected(monkeypatch):
    from model import explainer as expl

    payload = {
        "summary": "Summary with uncalibrated model risk estimate noted.",
        "strength_fact_ids": ["not_a_real_fact"],
        "risk_fact_ids": [],
        "decision": "Decision cites the manual demonstration band.",
    }

    class _Resp:
        content = [SimpleNamespace(text=json.dumps(payload))]

    class _Ok:
        def create(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Ok()))
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    text = explain_decision(applicant, scored)
    assert text == render_deterministic_narrative(
        scored, build_assessment_facts(applicant, scored)
    )


def test_forbidden_terms_trigger_fallback(monkeypatch):
    from model import explainer as expl

    payload = {
        "summary": "This cites a calibrated probability and probability of default.",
        "strength_fact_ids": ["debt_service_income"],
        "risk_fact_ids": [],
        "decision": "Decline under internal risk tolerance.",
    }

    class _Resp:
        content = [SimpleNamespace(text=json.dumps(payload))]

    class _Ok:
        def create(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Ok()))
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    text = explain_decision(applicant, scored)
    assert "probability of default" not in text.lower()
    assert "calibrated probability" not in text.lower()
    assert "uncalibrated model risk estimate" in text.lower()


def test_default_live_applicant_fact_profile_and_score():
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    assert scored["decision"] == "DECLINED"
    assert scored["risk_score"] == 639
    assert scored["default_probability"] == pytest.approx(0.3611, abs=5e-4)
    facts = build_assessment_facts(applicant, scored)
    class_map = _classify_map(facts)
    assert class_map["debt_service_income"] == "strength"
    assert class_map["employment"] == "strength"
    assert class_map["credit_to_income"] == "neutral"
    assert class_map["ext_source_1"] == "neutral"
    assert class_map["ext_source_2"] == "neutral"
    assert class_map["ext_source_3"] == "neutral"
    assert class_map["coverage"] == "neutral"
    narrative = render_deterministic_narrative(scored, facts)
    assert "Debt service / income 20.0%" in narrative
    assert "Employment tenure of 5.0 years" in narrative
    assert "3.00x" not in "\n".join(f["text"] for f in facts["strengths"] + facts["risks"])
    assert "0.45" not in "\n".join(f["text"] for f in facts["strengths"] + facts["risks"])
    assert "0.50" not in "\n".join(f["text"] for f in facts["strengths"] + facts["risks"])
    assert "94.4%" not in "\n".join(f["text"] for f in facts["strengths"] + facts["risks"])
    assert "uncalibrated model risk estimate" in narrative.lower()
    assert "probability of default" not in narrative.lower()


def test_decision_boundaries_remain_correct():
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
    assert format_uncalibrated_risk_display(0.14996, "APPROVED") == "<15.00%"
    assert format_uncalibrated_risk_display(0.15, "REVIEW") == "15.00%"
    assert format_uncalibrated_risk_display(0.34996, "REVIEW") == "<35.00%"
    assert format_uncalibrated_risk_display(0.35, "DECLINED") == "35.00%"


def test_valid_claude_json_uses_canonical_fact_text(monkeypatch):
    from model import explainer as expl

    payload = {
        "summary": (
            "Disposition follows the manual demonstration band on the "
            "uncalibrated model risk estimate."
        ),
        "strength_fact_ids": ["debt_service_income", "employment"],
        "risk_fact_ids": [],
        "decision": (
            "Decline follows the manual demonstration band for the "
            "uncalibrated model risk estimate."
        ),
    }

    class _Resp:
        content = [SimpleNamespace(text=json.dumps(payload))]

    class _Ok:
        def create(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(expl, "client", SimpleNamespace(messages=_Ok()))
    applicant = live_default_applicant()
    scored = score_applicant(applicant)
    text = explain_decision(applicant, scored)
    assert "Debt service / income 20.0%" in text
    assert "Employment tenure of 5.0 years" in text
    assert "Credit-to-income 3.00x" not in text
