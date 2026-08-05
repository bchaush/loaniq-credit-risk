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

LoanIQ shows one path: shared preprocessing → uncalibrated model risk estimate → manual demonstration band → deterministic fact engine → optional Claude selection of approved strength/risk fact IDs, with all displayed narrative text rendered locally.

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
- **Full assessment** scores first, then builds deterministic facts.
- Claude may select only approved strength and risk fact IDs. All displayed Summary, Decision, score, percentage, threshold, and bullet text is rendered locally from deterministic application facts.
- Neutral facts are never selectable. On API failure, malformed JSON, unknown IDs, forbidden terminology, or empty responses, a **deterministic local narrative** is used. Scoring is unchanged.

Global driver ranks in the UI are **training-level metadata order**, not applicant-level SHAP and not individualized causal reasons.

### Employment-to-age ratio (feature contract)

The deployed application preserves the feature contract used by the current serving artifacts. Historical training SQL computed this ratio from source day values (`DAYS_EMPLOYED` / `DAYS_BIRTH`), while the interactive and batch application derives it from the supplied `employed_years` and `age_years` fields (which may already be rounded for the application feature contract). The two formulas agree for the default applicant and covered parity fixtures; exact identity for every theoretical raw record must not be claimed. Changing this serving feature definition would require retraining, held-out metric evaluation, golden prediction comparison, and artifact verification.

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
the uncalibrated model risk estimate (36.11%; High Risk). Threshold band:
Approve: <15%; Review: ≥15% and <35%; Decline: ≥35% (manual demonstration
band). Observed profile indicators listed above are not individualized
model-attribution claims.
```

Neutral items such as credit-to-income **3.00x**, alternative composites **0.45 / 0.50**, and collateral coverage ratio **94.4%** are omitted from Strengths and Key Risks.

---

## Tech stack

| Layer | Tools | Purpose |
| --- | --- | --- |
| Data | SQLite, Pandas, Home Credit `application_train.csv` | Single-table feature view |
| Model | XGBoost native JSON booster + joblib preprocessing | Tabular inference with train-only medians / category maps |
| Explainability | Deterministic fact engine + optional Anthropic Claude fact-ID selection | Local Summary/Decision/bullets with local fallback |
| App | Streamlit | Single / batch / model tabs |
| Deploy | Streamlit Cloud | Demo hosting — set **Python 3.12** in Advanced settings |

---

## Runtime & artifact compatibility

Pinned direct dependencies are in `requirements.txt`. **CI is tested on Python 3.12.3**. Deployment requires **Python 3.12.x** in Streamlit Advanced settings; this prototype does not claim every 3.12 patch release was separately tested.

- Inference uses `model/loaniq_booster.json` (native XGBoost), exported with **exact** prediction parity against `model/loaniq_model.pkl`.
- `model/loaniq_model.pkl` is retained as the pickle reference; hashes and sizes live in `model/artifact_manifest.json` and are enforced by verification.
- `model/preprocessing.pkl` is the canonical runtime preprocessing artifact. `model/preprocessing.json` is an inspection-only sidecar of medians / category maps.
- `scripts/verify_artifact_compatibility.py` loads artifacts, fails on `InconsistentVersionWarning` / XGBoost pickle compatibility warnings, validates the manifest, and checks a **deterministic five-applicant golden set** covering:
  - **Decline** (default applicant: risk score **639**, uncalibrated estimate ≈ **36.11%**)
  - **Approve**
  - **Review**
  - unseen categorical values
  - missing numeric values filled with training medians

### Batch scoring

Batch and single-applicant paths share the same preprocessing and model. Equivalent results require equivalent model features. When source fields are present, engineered ratios/flags are validated against canonical SQL-aligned formulas.

- **Missing required columns** are rejected.
- **Missing optional model inputs** may use documented training medians or unknown-category handling; incomplete rows may differ from a fully populated single-applicant profile.
- **Malformed values**, **conflicting engineered fields**, and **reserved result columns** are rejected.

Exported batch columns use **uncalibrated model risk estimate** (raw + boundary-safe display); internal scoring keys are not part of the user-facing export.

Uploads are also rejected when they contain:
- invalid binary flags (must be exactly 0 or 1)
- fractional or negative count fields (`CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `credit_inquiries_year`)
- reserved scoring-output columns (`decision`, `risk_score`, `risk_tier`, `uncalibrated_model_risk_estimate`, `uncalibrated_model_risk_estimate_display`, `default_probability`), matched case-insensitively after trimming whitespace

Do **not** enter real personal information in the demo.

### Training a future serving bundle

Command (after database features are available):

```bash
python model/train.py
# optional provenance:
python model/train.py --training-commit "$(git rev-parse HEAD)"
# or: LOANIQ_TRAINING_COMMIT=<sha> python model/train.py
```

`model/train.py` calls `model.artifact_bundle.write_serving_bundle` after successful training and held-out evaluation. Supported published files:

| File | Role |
| --- | --- |
| `loaniq_booster.json` | **Runtime inference** (native XGBoost) |
| `preprocessing.pkl` | **Runtime** medians / category maps |
| `loaniq_model.pkl` | Pickle **reference** (parity / compatibility) |
| `metadata.json` | Metrics, feature order, demonstration bands |
| `artifact_manifest.json` | Hashes, sizes, runtime versions, training commit |
| `preprocessing.json` | **Inspection-only** sidecar (not loaded at runtime) |

`encoders.pkl` is a **legacy** committed artifact superseded by maps inside `preprocessing.pkl`. The training workflow does **not** update it.

Publication model: artifacts are written to a staging directory, hashed and validated, then published with rollback protection. Each file replacement uses an atomic filesystem operation (`os.replace`); the complete multi-file bundle is **not** claimed to be one indivisible filesystem transaction.

Training commit provenance (recorded in newly generated manifests only):
1. `--training-commit` CLI argument, else
2. `LOANIQ_TRAINING_COMMIT` environment variable, else
3. `git rev-parse HEAD` when available, else
4. `null` (training does not fail solely for missing Git metadata). Dirty working-tree status is recorded when safely detectable.

A changed feature definition — including employment-to-age derivation — requires retraining, held-out metric evaluation, golden prediction comparison, and artifact verification.

This maintenance patch does **not** retrain or modify the currently deployed model; live artifact hashes remain those recorded in `model/artifact_manifest.json`.

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
