# LoanIQ — Credit Risk Intelligence

> End-to-end ML underwriting platform: feature engineering → XGBoost → LLM explainability

**[Live Demo →](https://loaniq-credit-risk-fvx6dhozuvkixakbyaqfus.streamlit.app)**

---

## What it does
LoanIQ scores loan applicants using a trained XGBoost model and generates plain-English 
decision explanations via Claude AI — mimicking a production underwriting workflow.

## How it works
1. **Feature engineering** — 34 features derived from raw applicant data via SQL pipeline
2. **ML scoring** — XGBoost classifier (ROC-AUC: 0.7635) trained on 246K Home Credit samples
3. **LLM explanation** — Claude AI generates a 3-paragraph rationale per decision

## Tech stack
| Layer | Tools |
|---|---|
| Data | SQLite, Pandas, Home Credit dataset |
| Model | XGBoost, scikit-learn, joblib |
| Explainability | Anthropic Claude API |
| App | Streamlit |
| Deploy | Streamlit Cloud |

## Key results
- ROC-AUC: **0.7635** · PR-AUC: 0.2502
- 246,008 training samples · 8.1% default rate
- Decision thresholds: Approved < 15% · Review 15–35% · Declined > 35%