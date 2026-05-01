import anthropic
import sqlite3
import joblib
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# ── Load model artifacts ──────────────────────────────────────────
model    = joblib.load("model/loaniq_model.pkl")
encoders = joblib.load("model/encoders.pkl")

with open("model/metadata.json") as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["features"]

CAT_COLS = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    client = None
else:
    client = anthropic.Anthropic(api_key=api_key)


def encode_applicant(applicant: dict) -> np.ndarray:
    """Turn a raw applicant dict into the model's feature vector."""
    df = pd.DataFrame([applicant])
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_ else -1
            )
    df = df.fillna(0)
    # Align columns to training order
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_NAMES].values


def score_applicant(applicant: dict) -> dict:
    """Return risk score + probability for a single applicant."""
    X = encode_applicant(applicant)
    prob = model.predict_proba(X)[0][1]

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
        "default_probability": round(float(prob), 4),
        "risk_score": round((1 - prob) * 1000),   # 0–1000, higher = better
        "decision":   decision,
        "risk_tier":  risk_tier
    }


def explain_decision(applicant: dict, score_result: dict) -> str:
    """Call Claude API to generate plain-English explanation."""
    if client is None:
        return "AI explanation unavailable (missing API key)."
    # Pull key risk factors for the prompt
    prob    = score_result["default_probability"]
    dti     = applicant.get("debt_to_income", "N/A")
    a2i     = applicant.get("annuity_to_income", "N/A")
    ext2    = applicant.get("EXT_SOURCE_2", "N/A")
    ext3    = applicant.get("EXT_SOURCE_3", "N/A")
    age     = applicant.get("age_years", "N/A")
    emp     = applicant.get("employed_years", "N/A")
    unemp   = applicant.get("is_unemployed", 0)
    ltv     = applicant.get("ltv_ratio", "N/A")
    low2    = applicant.get("low_ext_score_2", 0)
    low3    = applicant.get("low_ext_score_3", 0)
    edu     = applicant.get("NAME_EDUCATION_TYPE", "N/A")
    income  = applicant.get("NAME_INCOME_TYPE", "N/A")
    inq     = applicant.get("credit_inquiries_year", 0)

    prompt = f"""You are an internal underwriting analyst at LoanIQ 
documenting a credit decision for internal review.
Do not address the applicant directly.
Do not use "Thank you", "we appreciate", "we recommend",
"a member of our team", or any customer-facing language.
Do not use "your application" as the subject.
Write for a senior credit risk manager reviewing 
an internal memo.
Do not use HTML tags or markdown code blocks.
Do not mention EXT_SOURCE, column names, or 
internal variable names.

Use this exact output structure and nothing else:

UNDERWRITING RATIONALE

Summary
[1-2 sentences — internal analyst tone, 
use "the applicant" not "you"]

Strengths
- [bullet]
- [bullet]
- [bullet]

Key Risks
- [bullet]
- [bullet]
- [bullet]

Decision
[1-2 sentences explaining disposition — 
use "the applicant" not "you", no customer language]

Do not add any sections beyond these four.
Do not add any closing remarks or sign-off.

APPLICATION DATA:
- Decision: {score_result['decision']} ({score_result['risk_tier']})
- Default Probability: {prob:.1%}
- Risk Score: {score_result['risk_score']} / 1000
- Debt-to-Income Ratio: {dti}
- Annuity-to-Income Ratio: {a2i}
- External Credit Score 2: {ext2} (flag: {'below threshold' if low2 else 'acceptable'})
- External Credit Score 3: {ext3} (flag: {'below threshold' if low3 else 'acceptable'})
- Age: {age} years
- Employment: {'Currently unemployed' if unemp else f'{emp} years employed'}
- Education: {edu}
- Income Type: {income}
- Loan-to-Value Ratio: {ltv}
- Credit Inquiries (last year): {inq}

Keep it under 150 words. Use the exact four-section 
structure above. Internal analyst tone throughout."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def full_assessment(applicant: dict) -> dict:
    """Score + explain in one call. This is what app.py will use."""
    result   = score_applicant(applicant)
    result["explanation"] = explain_decision(applicant, result)
    return result


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    test_applicant = {
        "AMT_INCOME_TOTAL":         45000,
        "AMT_CREDIT":               180000,
        "AMT_ANNUITY":              9000,
        "AMT_GOODS_PRICE":          170000,
        "debt_to_income":           4.0,
        "annuity_to_income":        0.20,
        "loan_term_implied":        20,
        "ltv_ratio":                0.94,
        "age_years":                34,
        "employed_years":           3.5,
        "employment_to_age_ratio":  0.10,
        "is_unemployed":            0,
        "EXT_SOURCE_1":             0.48,
        "EXT_SOURCE_2":             0.28,   # below 0.3 threshold → flag
        "EXT_SOURCE_3":             0.31,
        "ext_score_sum":            1.07,
        "low_ext_score_2":          1,
        "low_ext_score_3":          0,
        "CNT_CHILDREN":             1,
        "CNT_FAM_MEMBERS":          3,
        "FLAG_OWN_CAR":             0,
        "FLAG_OWN_REALTY":          1,
        "many_children":            0,
        "REGION_RATING_CLIENT":     2,
        "REG_CITY_NOT_WORK_CITY":   0,
        "FLAG_DOCUMENT_3":          1,
        "credit_inquiries_year":    2,
        "high_inquiry_flag":        0,
        "NAME_INCOME_TYPE":         "Working",
        "NAME_EDUCATION_TYPE":      "Secondary / secondary special",
        "NAME_FAMILY_STATUS":       "Married",
        "NAME_HOUSING_TYPE":        "House / apartment",
        "OCCUPATION_TYPE":          "Laborers",
        "ORGANIZATION_TYPE":        "Business Entity Type 3",
    }

    print("Running full assessment...\n")
    result = full_assessment(test_applicant)

    print(f"Decision:     {result['decision']}")
    print(f"Risk Tier:    {result['risk_tier']}")
    print(f"Default Prob: {result['default_probability']:.1%}")
    print(f"Risk Score:   {result['risk_score']} / 1000")
    print(f"\n--- EXPLANATION ---\n{result['explanation']}")