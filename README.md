![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-ECB900?style=flat-square)
![Claude](https://img.shields.io/badge/Anthropic-Claude_API-D4A574?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

# LoanIQ — Credit Risk Intelligence

**Research / portfolio prototype:** XGBoost scoring with a fact-bounded Claude narrative. This is **not** a production underwriting system and is **not** a regulated adverse-action system.

---

**[▶ Launch live demo](https://loaniq-credit-risk-3pteppifk5xqkxxuqnq7xq.streamlit.app/)**

---

## The problem

Most credit demos stop at a score. Stakeholders still need a readable rationale that cannot invent strengths, risks, or causal claims.

LoanIQ shows one path: shared preprocessing → uncalibrated model risk estimate → manual demonstration band → deterministic fact engine → optional Claude connective prose that is validated before display.

---

## How it works

**Step 1 — Feature engineering:** **34** features from Home Credit **`application_train.csv`** only (SQLite `applications` → `model_features`). No bureau / installment / balance side tables.

**Step 2 — ML scoring:** **XGBoost** with stratified train / validation / held-out test splits. Reported metrics are **test-only** (from `model/metadata.json`):
- **ROC-AUC:** **0.7626**
- **PR-AUC:** **0.2500**
- **Train:** **184,506** · **Validation:** **61,502** · **Test:** **61,503**
- Served trees: **289** (`best_iteration` **288**)

The score is an **uncalibrated model risk estimate**. Manual demonstration bands:

- **Approve:** &lt;15%
- **Review:** ≥15% and &lt;35%
- **Decline:** ≥35%

These are **not** validation-tuned cutoffs and **not** fitted probability calibration.

**Step 3 — Narrative:**
- **Quick score** never calls Anthropic.
- **Full assessment** scores first, then builds deterministic facts. Claude receives only approved strength/risk fact IDs plus decision context — not neutral metrics.
- Claude JSON is parsed and validated; Strengths / Key Risks are assembled from **canonical local fact text**.
- On API failure, malformed JSON, unknown IDs, forbidden terminology, or empty responses, a **deterministic local narrative** is used. Scoring is unchanged.

Global driver ranks in the UI are **training-level metadata order**, not applicant-level SHAP and not individualized causal reasons.

---

## Sample output *(illustrative; fact-bounded)*

```
Summary:
The applicant receives a DECLINED disposition (High Risk) under the
manual demonstration band. The uncalibrated model risk estimate is 36.11%.

Strengths:
- Debt service / income 20.0% indicates a manageable repayment burden
  (observed profile indicator).
- Employment tenure of 5.0 years supports income continuity
  (observed profile indicator).

Key Risks:
- None identified.

Decision:
The DECLINED outcome follows the manual demonstration band applied to
the uncalibrated model risk estimate (36.11%). Observed profile indicators
listed above are not individualized model-attribution claims.
```

Neutral items such as credit-to-income **3.00x**, alternative composites **0.45 / 0.50**, and collateral coverage ratio **94.4%** are omitted from Strengths and Key Risks.

---

## Tech stack

| Layer | Tools | Purpose |
| --- | --- | --- |
| Data | SQLite, Pandas, Home Credit `application_train.csv` | Single-table feature view |
| Model | XGBoost native JSON booster + joblib preprocessing | Tabular inference with train-only medians / category maps |
| Explainability | Deterministic fact engine + optional Anthropic Claude | Validated narrative with local fallback |
| App | Streamlit | Single / batch / model tabs |
| Deploy | Streamlit Cloud | Demo hosting — set **Python 3.12** in Advanced settings |

---

## Runtime & artifact compatibility

Pinned direct dependencies are in `requirements.txt`. **CI is tested on Python 3.12.3**. Deployment requires **Python 3.12.x** in Streamlit Advanced settings; this prototype does not claim every 3.12 patch release was separately tested.

- Inference uses `model/loaniq_booster.json` (native XGBoost), exported with **exact** prediction parity against `model/loaniq_model.pkl`.
- `model/loaniq_model.pkl` is retained as the pickle reference; hashes and sizes live in `model/artifact_manifest.json` and are enforced by verification.
- `scripts/verify_artifact_compatibility.py` loads artifacts, fails on `InconsistentVersionWarning` / XGBoost pickle compatibility warnings, validates the manifest, and checks a **deterministic five-applicant golden set** covering:
  - **Decline** (default applicant: risk score **639**, uncalibrated estimate ≈ **36.11%**)
  - **Approve**
  - **Review**
  - unseen categorical values
  - missing numeric values filled with training medians

### Batch scoring

Batch and single-applicant paths share the same preprocessing and model. Equivalent results require equivalent model features. When source fields are present, engineered ratios/flags are validated against canonical SQL-aligned formulas. Absent optional features use training medians or unknown-category codes, so incomplete rows may differ from a fully populated single-applicant profile. Exported batch columns use **uncalibrated model risk estimate** (raw + boundary-safe display); internal scoring keys are not part of the user-facing export.

Do **not** enter real personal information in the demo.

---

## Repo structure

```
loaniq-credit-risk/
├── app.py                 # Streamlit underwriting workspace
├── requirements.txt       # Pinned runtime dependencies
├── runtime.txt            # python-3.12.3
├── README.md              # Project overview (this file)
├── scripts/               # Artifact compatibility verification
├── database/              # SQLite build + feature view scripts
├── sql/                   # Feature-engineering SQL
├── model/                 # Train, preprocess, explainer, artifacts, manifest
└── tests/                 # Integrity + narrative tests
```

---

## Limitations

**Research prototype only** — not production underwriting and not a claim of regulatory compliance (Basel, FCRA, ECOA, or similar). Home Credit data is an international research proxy. A live US deployment would need regulated data, fair-lending review, adverse-action governance, monitoring, and independently validated reason codes. See in-app compliance / ECOA notes.

---

## About the author

Built by **Bora Chaush** — MS Business Analytics @ Brandeis International Business School. Background in finance, accounting (**PwC**), and ML engineering. [Connect on LinkedIn](https://www.linkedin.com/in/bora-chaush-msba/).
