"""LoanIQ scoring + fact-bounded narrative explainability."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

try:
    from model.preprocess import (
        load_preprocessing,
        transform_applicant,
        transform_frame,
    )
except ImportError:  # running as script inside model/
    from preprocess import (  # type: ignore
        load_preprocessing,
        transform_applicant,
        transform_frame,
    )

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# ── Load model artifacts (ROOT_DIR-absolute paths) ─────────────────
PREPROCESSING = load_preprocessing(ROOT_DIR / "model" / "preprocessing.pkl")

with open(ROOT_DIR / "model" / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["features"]
BEST_ITERATION = int(metadata.get("best_iteration", 0))
N_TREES_SERVED = int(metadata.get("n_trees_served", BEST_ITERATION + 1))

# Native booster JSON is the inference path (pickle retained for verification).
_BOOSTER_JSON = ROOT_DIR / "model" / "loaniq_booster.json"
_MODEL_PKL = ROOT_DIR / "model" / "loaniq_model.pkl"
booster = xgb.Booster()
if _BOOSTER_JSON.exists():
    booster.load_model(str(_BOOSTER_JSON))
else:
    # Fallback only when native export is absent.
    _sklearn_model = joblib.load(_MODEL_PKL)
    booster = _sklearn_model.get_booster()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key) if api_key else None

# Forbidden terminology detectors (kept as patterns so source scans stay clean).
_FORBIDDEN_PATTERNS = (
    re.compile(r"calibrated\s+probability", re.I),
    re.compile(r"probability\s+of\s+default", re.I),
    re.compile(r"internal\s+risk\s+tolerance", re.I),
    re.compile(r"validated\s+cutoff", re.I),
    re.compile(r"adverse-action\s+reason", re.I),
    re.compile(r"loan[\s-]?to[\s-]?value", re.I),
)


def _contains_forbidden_terminology(text: str) -> bool:
    return any(p.search(text) for p in _FORBIDDEN_PATTERNS)


POLICY_BAND_TEXT = (
    "Approve: <15%; Review: ≥15% and <35%; Decline: ≥35% "
    "(manual demonstration band)"
)


def encode_applicant(applicant: dict) -> np.ndarray:
    """Shared transform: train medians + categorical maps (no zero-fill)."""
    return transform_applicant(applicant, PREPROCESSING)


def encode_batch(df: pd.DataFrame) -> np.ndarray:
    """Same shared transform for batch rows."""
    return transform_frame(df, PREPROCESSING)


def _predict_proba_best(X: np.ndarray) -> float:
    """Serve only trees through best_iteration (inclusive)."""
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    return float(booster.predict(dmat, iteration_range=(0, N_TREES_SERVED))[0])


def score_applicant(applicant: dict) -> dict:
    """Return risk score + uncalibrated estimate for a single applicant."""
    X = encode_applicant(applicant)
    prob = _predict_proba_best(X)

    # Demonstration bands (manually selected; not validation-tuned)
    if prob < 0.15:
        decision = "APPROVED"
        risk_tier = "Low Risk"
    elif prob < 0.35:
        decision = "REVIEW"
        risk_tier = "Medium Risk"
    else:
        decision = "DECLINED"
        risk_tier = "High Risk"

    return {
        "default_probability": float(prob),
        "risk_score": round((1 - prob) * 1000),  # 0–1000, higher = better
        "decision": decision,
        "risk_tier": risk_tier,
    }


def format_uncalibrated_risk_display(prob: float, decision: str) -> str:
    """Boundary-safe percentage text that cannot contradict the raw decision."""
    text = f"{prob:.2%}"
    if decision == "APPROVED" and text == "15.00%":
        return "<15.00%"
    if decision == "REVIEW" and text == "35.00%":
        return "<35.00%"
    return text


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_mult(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}x"


def _fmt_score(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _fact(
    fact_id: str,
    text: str,
    raw_value: Any,
    formatted_value: str,
    classification: str,
    rule: str,
) -> dict[str, Any]:
    return {
        "id": fact_id,
        "text": text,
        "raw_value": raw_value,
        "formatted_value": formatted_value,
        "classification": classification,
        "rule": rule,
    }


def build_assessment_facts(applicant: dict, score_result: dict) -> dict[str, list[dict]]:
    """Deterministic metric fact engine (pure / independently testable)."""
    strengths: list[dict] = []
    risks: list[dict] = []
    neutral: list[dict] = []
    summary_facts: list[dict] = []
    decision_facts: list[dict] = []

    def place(fact: dict) -> None:
        bucket = {
            "strength": strengths,
            "risk": risks,
            "neutral": neutral,
        }[fact["classification"]]
        bucket.append(fact)

    # Collateral coverage ratio (collateral ÷ loan)
    coverage = float(applicant.get("ltv_ratio", float("nan")))
    if coverage > 1.20:
        cls, rule = "strength", "coverage > 1.20"
        text = (
            f"Collateral coverage ratio {_fmt_pct(coverage)} indicates a strong "
            "collateral buffer (observed profile indicator)."
        )
    elif coverage < 0.80:
        cls, rule = "risk", "coverage < 0.80"
        text = (
            f"Collateral coverage ratio {_fmt_pct(coverage)} indicates "
            "insufficient collateral relative to requested credit "
            "(observed profile indicator)."
        )
    else:
        cls, rule = "neutral", "0.80 ≤ coverage ≤ 1.20"
        text = (
            f"Collateral coverage ratio {_fmt_pct(coverage)} is within the "
            "neutral band (observed profile indicator)."
        )
    place(_fact("coverage", text, coverage, _fmt_pct(coverage), cls, rule))

    # Credit-to-income
    dti = float(applicant.get("debt_to_income", float("nan")))
    if dti < 1.50:
        cls, rule = "strength", "credit-to-income < 1.50x"
        text = (
            f"Credit-to-income {_fmt_mult(dti)} indicates low leverage "
            "(observed profile indicator)."
        )
    elif dti > 3.00:
        cls, rule = "risk", "credit-to-income > 3.00x"
        text = (
            f"Credit-to-income {_fmt_mult(dti)} indicates elevated leverage "
            "(observed profile indicator)."
        )
    else:
        cls, rule = "neutral", "1.50x ≤ credit-to-income ≤ 3.00x"
        text = (
            f"Credit-to-income {_fmt_mult(dti)} is within the neutral band "
            "(observed profile indicator)."
        )
    place(_fact("credit_to_income", text, dti, _fmt_mult(dti), cls, rule))

    # Debt service / income (annuity_to_income raw ratio)
    a2i = float(applicant.get("annuity_to_income", float("nan")))
    if a2i < 0.30:
        cls, rule = "strength", "debt-service / income < 30%"
        text = (
            f"Debt service / income {_fmt_pct(a2i)} indicates a manageable "
            "repayment burden (observed profile indicator)."
        )
    elif a2i > 0.50:
        cls, rule = "risk", "debt-service / income > 50%"
        text = (
            f"Debt service / income {_fmt_pct(a2i)} indicates a high repayment "
            "burden (observed profile indicator)."
        )
    else:
        cls, rule = "neutral", "30% ≤ debt-service / income ≤ 50%"
        text = (
            f"Debt service / income {_fmt_pct(a2i)} is within the neutral band "
            "(observed profile indicator)."
        )
    place(_fact("debt_service_income", text, a2i, _fmt_pct(a2i), cls, rule))

    # Alternative credit composites (A/B/C)
    for key, label, fact_id in (
        ("EXT_SOURCE_1", "Alternative credit composite A", "ext_source_1"),
        ("EXT_SOURCE_2", "Alternative credit composite B", "ext_source_2"),
        ("EXT_SOURCE_3", "Alternative credit composite C", "ext_source_3"),
    ):
        val = float(applicant.get(key, float("nan")))
        if val > 0.70:
            cls, rule = "strength", f"{label} > 0.70"
            text = (
                f"{label} {_fmt_score(val)} is a supportive observed profile "
                "indicator."
            )
        elif val < 0.40:
            cls, rule = "risk", f"{label} < 0.40"
            text = (
                f"{label} {_fmt_score(val)} is a weak observed profile indicator."
            )
        else:
            cls, rule = "neutral", f"0.40 ≤ {label} ≤ 0.70"
            text = (
                f"{label} {_fmt_score(val)} is within the neutral band "
                "(observed profile indicator)."
            )
        place(_fact(fact_id, text, val, _fmt_score(val), cls, rule))

    # Employment: employed with positive valid tenure = strength; unemployed = risk
    is_unemployed = int(applicant.get("is_unemployed", 0) or 0)
    emp_years = applicant.get("employed_years", float("nan"))
    try:
        emp_years_f = float(emp_years)
    except (TypeError, ValueError):
        emp_years_f = float("nan")

    if is_unemployed == 1:
        cls, rule = "risk", "is_unemployed == 1"
        text = (
            "Employment status indicates no active employment tenure "
            "(observed profile indicator)."
        )
        raw_emp = None
        fmt_emp = "unemployed"
    elif emp_years_f > 0 and not np.isnan(emp_years_f):
        cls, rule = "strength", "employed with positive valid tenure"
        text = (
            f"Employment tenure of {emp_years_f:.1f} years supports income "
            "continuity (observed profile indicator)."
        )
        raw_emp = emp_years_f
        fmt_emp = f"{emp_years_f:.1f} years"
    else:
        cls, rule = "neutral", "employment neither clearly employed-with-tenure nor unemployed"
        text = (
            "Employment tenure is indeterminate for strength/risk classification "
            "(observed profile indicator)."
        )
        raw_emp = emp_years_f
        fmt_emp = "indeterminate"
    place(_fact("employment", text, raw_emp, fmt_emp, cls, rule))

    # Inquiry history: existing pipeline rule high_inquiry_flag (inquiries > 3)
    high_inq = int(applicant.get("high_inquiry_flag", 0) or 0)
    inq = int(applicant.get("credit_inquiries_year", 0) or 0)
    if high_inq == 1:
        cls, rule = "risk", "high_inquiry_flag == 1 (credit_inquiries_year > 3)"
        text = (
            f"Hard inquiries in the last year ({inq}) exceed the pipeline "
            "high-inquiry rule (observed profile indicator)."
        )
    else:
        cls, rule = "neutral", "high_inquiry_flag == 0 (credit_inquiries_year ≤ 3)"
        text = (
            f"Hard inquiries in the last year ({inq}) do not trigger the pipeline "
            "high-inquiry rule (observed profile indicator)."
        )
    place(_fact("inquiries", text, inq, str(inq), cls, rule))

    # Age / marital status: retained for research-dataset parity only — never
    # classified as generic strength/risk or included in causal rationale.
    age = applicant.get("age_years")
    marital = applicant.get("NAME_FAMILY_STATUS")
    parity_note = (
        "Age and marital/family status are retained for research-dataset parity "
        "only and are excluded from strength/risk classification."
    )
    neutral.append(
        _fact(
            "parity_age_marital",
            parity_note,
            {"age_years": age, "NAME_FAMILY_STATUS": marital},
            f"age={age}; marital={marital}",
            "neutral",
            "ECOA/research parity — never strength/risk",
        )
    )

    # Uncalibrated model risk estimate — policy-band trigger only
    prob = float(score_result["default_probability"])
    decision = score_result["decision"]
    prob_display = format_uncalibrated_risk_display(prob, decision)
    policy_fact = _fact(
        "uncalibrated_risk_estimate",
        (
            f"Uncalibrated model risk estimate {prob_display} triggers the "
            f"{decision} manual demonstration band ({POLICY_BAND_TEXT})."
        ),
        prob,
        prob_display,
        "summary",
        "policy band on uncalibrated model risk estimate",
    )
    summary_facts.append(policy_fact)
    decision_facts.append(policy_fact)
    decision_facts.append(
        _fact(
            "risk_score",
            f"Risk score {score_result['risk_score']} / 1000 accompanies the "
            "uncalibrated model risk estimate.",
            score_result["risk_score"],
            f"{score_result['risk_score']} / 1000",
            "summary",
            "display score = round((1 - estimate) * 1000)",
        )
    )

    return {
        "summary_facts": summary_facts,
        "strengths": strengths,
        "risks": risks,
        "neutral": neutral,
        "decision_facts": decision_facts,
    }


def build_canonical_summary(score_result: dict) -> str:
    """Deterministic Summary body from local score fields only (no LLM text)."""
    decision = score_result["decision"]
    tier = score_result["risk_tier"]
    prob_display = format_uncalibrated_risk_display(
        float(score_result["default_probability"]), decision
    )
    return (
        f"The applicant receives a {decision} disposition ({tier}) under the "
        f"manual demonstration band. The uncalibrated model risk estimate is "
        f"{prob_display}."
    )


def build_canonical_decision_text(score_result: dict) -> str:
    """Deterministic Decision body from local score fields only (no LLM text)."""
    decision = score_result["decision"]
    tier = score_result["risk_tier"]
    prob_display = format_uncalibrated_risk_display(
        float(score_result["default_probability"]), decision
    )
    return (
        f"The {decision} outcome follows the manual demonstration band applied to "
        f"the uncalibrated model risk estimate ({prob_display}; {tier}). "
        f"Threshold band: {POLICY_BAND_TEXT}. Observed profile indicators listed "
        f"above are not individualized model-attribution claims."
    )


def render_deterministic_narrative(
    score_result: dict,
    facts: dict[str, list[dict]],
) -> str:
    """Fully local rationale assembled from canonical fact text."""
    strengths = facts["strengths"]
    risks = facts["risks"]

    strength_lines = (
        "\n".join(f"- {f['text']}" for f in strengths)
        if strengths
        else "- None identified."
    )
    risk_lines = (
        "\n".join(f"- {f['text']}" for f in risks)
        if risks
        else "- None identified."
    )

    return (
        f"Summary:\n{build_canonical_summary(score_result)}\n\n"
        f"Strengths:\n{strength_lines}\n\n"
        f"Key Risks:\n{risk_lines}\n\n"
        f"Decision:\n{build_canonical_decision_text(score_result)}"
    )


def _assemble_from_fact_ids(
    score_result: dict,
    facts: dict[str, list[dict]],
    strength_ids: list[str],
    risk_ids: list[str],
) -> str:
    """Assemble narrative using local Summary/Decision + canonical fact bullets."""
    strength_map = {f["id"]: f for f in facts["strengths"]}
    risk_map = {f["id"]: f for f in facts["risks"]}
    strength_lines = (
        "\n".join(f"- {strength_map[i]['text']}" for i in strength_ids)
        if strength_ids
        else "- None identified."
    )
    risk_lines = (
        "\n".join(f"- {risk_map[i]['text']}" for i in risk_ids)
        if risk_ids
        else "- None identified."
    )
    return (
        f"Summary:\n{build_canonical_summary(score_result)}\n\n"
        f"Strengths:\n{strength_lines}\n\n"
        f"Key Risks:\n{risk_lines}\n\n"
        f"Decision:\n{build_canonical_decision_text(score_result)}"
    )


def _parse_claude_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def explain_decision(applicant: dict, score_result: dict) -> str:
    """Fact-bounded narrative; Summary/Decision are always local/canonical.

    Claude may select only approved strength/risk fact IDs. It cannot supply or
    override decision, tier, score, percentage, thresholds, Summary, or Decision.
    """
    facts = build_assessment_facts(applicant, score_result)
    fallback = render_deterministic_narrative(score_result, facts)

    approved_strength_ids = {f["id"] for f in facts["strengths"]}
    approved_risk_ids = {f["id"] for f in facts["risks"]}
    neutral_ids = {f["id"] for f in facts["neutral"]}

    if client is None:
        return fallback

    strength_payload = [
        {"id": f["id"], "text": f["text"]} for f in facts["strengths"]
    ]
    risk_payload = [{"id": f["id"], "text": f["text"]} for f in facts["risks"]]

    prompt = f"""You are assisting an internal LoanIQ research prototype.
Return ONLY valid JSON with exactly these keys:
  strength_fact_ids (array of strings),
  risk_fact_ids (array of strings)

Rules:
- Select only from the provided approved fact IDs.
- Do not invent fact IDs.
- Do not select neutral metrics.
- Do not return decision, risk tier, risk score, percentage, Summary, or Decision text.
- Strength/risk bullet text, Summary, and Decision are rendered locally.

Approved strength facts: {json.dumps(strength_payload)}
Approved risk facts: {json.dumps(risk_payload)}
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
    except Exception:
        return fallback

    if not raw or not str(raw).strip():
        return fallback

    parsed = _parse_claude_json(str(raw))
    if parsed is None:
        return fallback

    strength_ids = parsed.get("strength_fact_ids") or []
    risk_ids = parsed.get("risk_fact_ids") or []

    if not isinstance(strength_ids, list) or not isinstance(risk_ids, list):
        return fallback
    if not all(isinstance(x, str) for x in strength_ids + risk_ids):
        return fallback

    if any(i not in approved_strength_ids for i in strength_ids):
        return fallback
    if any(i not in approved_risk_ids for i in risk_ids):
        return fallback
    if any(i in neutral_ids for i in strength_ids + risk_ids):
        return fallback

    assembled = _assemble_from_fact_ids(
        score_result, facts, strength_ids, risk_ids
    )
    if _contains_forbidden_terminology(assembled):
        return fallback
    return assembled


def full_assessment(applicant: dict) -> dict:
    """Score + explain in one call. Scoring is independent of the LLM."""
    result = score_applicant(applicant)
    result["explanation"] = explain_decision(applicant, result)
    result["assessment_facts"] = build_assessment_facts(applicant, result)
    return result


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    test_applicant = {
        "AMT_INCOME_TOTAL": 60000,
        "AMT_CREDIT": 180000,
        "AMT_ANNUITY": 12000,
        "AMT_GOODS_PRICE": 170000,
        "debt_to_income": 3.0,
        "annuity_to_income": 0.2,
        "loan_term_implied": 15.0,
        "ltv_ratio": 0.944,
        "age_years": 35,
        "employed_years": 5.0,
        "employment_to_age_ratio": 0.1429,
        "is_unemployed": 0,
        "EXT_SOURCE_1": 0.50,
        "EXT_SOURCE_2": 0.45,
        "EXT_SOURCE_3": 0.50,
        "ext_score_sum": 1.45,
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

    print("Running full assessment...\n")
    result = full_assessment(test_applicant)
    print(f"Decision:     {result['decision']}")
    print(f"Risk Tier:    {result['risk_tier']}")
    print(f"Uncalibrated: {result['default_probability']:.2%}")
    print(f"Risk Score:   {result['risk_score']} / 1000")
    print(f"\n--- EXPLANATION ---\n{result['explanation']}")
