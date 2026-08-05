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

Most credit models stop at a score. That is not enough when stakeholders need a readable rationale.

This prototype shows how scoring and an LLM memo can sit on one path. It is **not** a production underwriting system and does **not** claim Basel, FCRA, or ECOA compliance.

---

## How it works

**Step 1 — Feature engineering:** **34** features from the Home Credit **`application_train`** table via SQLite (`applications` → `model_features`). Ratios, tenure, bureau-style composites, and categoricals used by the classifier.

**Step 2 — ML scoring:** **XGBoost** with train / validation / held-out test discipline. Reported metrics are **test-only**. Decision bands (**&lt;15%** / **15–35%** / **&gt;35%**) are **manually selected demonstration policy bands**, not validation-tuned cutoffs.

**Step 3 — LLM explainability:** **Claude** drafts an internal-style memo from application fields. Global gain bars in the UI are training-level drivers, not per-applicant SHAP.

---

## Sample output *(illustrative)*

```
Summary:
Declined — model-estimated default probability sits above the demo decline band.

Strengths:
- Employment tenure supports income continuity.

Key Risks:
- Debt-to-income elevated versus stated income.
- Alternative bureau composites below internal demo floors.

Decision:
Decline follows the demonstration band given leverage and bureau composites.
```

---

## Tech stack

| Layer | Tools | Purpose |
| --- | --- | --- |
| Data | SQLite, Pandas, Home Credit application table | Lightweight feature view without a heavy warehouse |
| Model | XGBoost, scikit-learn, joblib | Tabular model with class imbalance handling; fast inference |
| Explainability | Anthropic Claude API | Readable rationales (prototype; not regulated adverse-action text) |
| App | Streamlit | Interactive single / batch / model tabs |
| Deploy | Streamlit Cloud | Demo hosting |

---

## Key results

Metrics are written to `model/metadata.json` after training (test split only). Check that file for current **ROC-AUC**, **PR-AUC**, and sample counts.

- ✅ Decision bands (demo): **&lt;15%** approve · **15–35%** review · **&gt;35%** decline
- ✅ Preprocessing fitted on **train only**; early stopping on **validation**; final report on **test**

---

## Repo structure

```
loaniq-credit-risk/
├── app.py                 # Streamlit underwriting workspace
├── requirements.txt       # Runtime dependencies
├── README.md              # Project overview (this file)
├── database/              # SQLite build + feature view scripts
├── sql/                   # Feature-engineering SQL
├── model/                 # Train, preprocess, explainer, artifacts
└── tests/                 # Integrity tests
```

---

## Limitations

US Fintech **prototype**. Home Credit data is an international research proxy. Production use would need regulated data, fair-lending review, adverse-action governance, and monitoring. See in-app compliance / ECOA notes.

---

## About the author

Built by **Bora Chaush** — MS Business Analytics @ Brandeis International Business School. Background in finance, accounting (**PwC**), and ML engineering. [Connect on LinkedIn](https://www.linkedin.com/in/bora-chaush-90b257239).
