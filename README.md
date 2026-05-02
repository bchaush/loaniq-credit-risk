![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-ECB900?style=flat-square)
![Claude](https://img.shields.io/badge/Anthropic-Claude_API-D4A574?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

# LoanIQ — Credit Risk Intelligence

**End-to-end ML underwriting platform with XGBoost scoring and Claude AI explainability.**

---

**[▶ Launch live demo](https://loaniq-credit-risk-fvx6dhozuvkixakbyaqfus.streamlit.app)**

---

## The problem

Most production credit models stop at a probability or score. That is not enough for policy teams or regulators.

Basel III capital frameworks and FCRA-style adverse-action rules expect transparent reasons when credit is denied or priced aggressively. LoanIQ wires explainability into the same path as the score—so risk and narrative stay aligned.

---

## How it works

Step 1 — Feature engineering: 34 features engineered from 5 raw Home Credit tables through a SQLite pipeline—bureau history, application attributes, installment behavior, and balance dynamics.

Step 2 — ML scoring: XGBoost classifier on 246,008 training rows (8.1% default rate). ROC-AUC 0.7635, PR-AUC 0.2502. Three bands: Approved default prob &lt;15% · Review 15–35% · Declined &gt;35%.

Step 3 — LLM explainability: Claude API drafts a structured underwriting memo per decision—risk drivers, offsets, and why the band triggered.

---

## Sample output *(illustrative)*

```
Summary:
Declined — default probability sits above policy tolerance given leverage and bureau weakness.

Strengths:
- Employment tenure (4.5 years) supports income continuity.

Key Risks:
- Debt-to-income at 4.2× signals stretched leverage versus stated income.
- Alternative bureau composites B/C sit below internal floors (flags raised).
- Debt service / income near upper band leaves little cushion for shocks.

Decision:
Decline aligns with elevated default probability and thin mitigants on affordability; adverse-action reasons would cite leverage and bureau composites.
```

---

## Tech stack

| Layer | Tools | Purpose |
| --- | --- | --- |
| Data | SQLite, Pandas, Home Credit | Lightweight relational feature pipeline without heavy DB ops |
| Model | XGBoost, scikit-learn, joblib | Strong tabular performance with **class imbalance**; fast batch inference |
| Explainability | Anthropic Claude API | Readable rationales without maintaining template libraries |
| App | Streamlit | Rapid interactive underwriting UI |
| Deploy | Streamlit Cloud | Zero-ops hosting aligned with demo workflows |

---

## Key results

- ROC-AUC: 0.7635
- PR-AUC: 0.2502
- Training n: 246,008 · Test n: 61,503 · Default rate: 8.1%
- Thresholds: &lt;15% approve · 15–35% manual review · &gt;35% decline

---

## Repo structure

```
loaniq-credit-risk/
├── app.py                 # Streamlit underwriting workspace (single / batch / model tabs)
├── requirements.txt       # Runtime dependencies
├── README.md              # Project overview (this file)
├── database/              # SQLite build + feature application scripts
├── sql/                   # Feature-engineering SQL (Home Credit → training schema)
└── model/                 # Training script, Claude explainer, metadata, serialized model artifacts
```

---

## About the author

Built by Bora Chaush — MS Business Analytics @ Brandeis International Business School. Background in finance, accounting (PwC), and ML engineering. [Connect on LinkedIn](https://www.linkedin.com/in/bora-chaush-90b257239).
