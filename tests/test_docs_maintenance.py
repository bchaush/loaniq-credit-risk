"""Documentation and terminology integrity for maintenance wording."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

README = (ROOT / "README.md").read_text(encoding="utf-8")
SQL = (ROOT / "sql" / "feature_engineering.sql").read_text(encoding="utf-8")
PREPROCESS = (ROOT / "model" / "preprocess.py").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_readme_claude_is_fact_id_selection_only():
    assert "Claude connective prose" not in README
    assert "Claude-generated Summary" not in README
    assert "approved strength and risk fact IDs" in README or "approved `strength_fact_ids`" in README
    assert "Summary" in README and "Decision" in README
    assert "rendered locally" in README.lower() or "rendered locally from deterministic" in README


def test_readme_bundle_wording_is_rollback_not_transaction():
    assert "published atomically" not in README.lower()
    assert "rollback" in README.lower()
    assert "staging" in README.lower()
    assert "not" in README.lower() and "indivisible filesystem transaction" in README.lower()


def test_readme_batch_required_vs_optional():
    assert "Missing required columns" in README
    assert "Missing optional model inputs" in README
    assert "reserved result columns" in README.lower() or "reserved scoring-output" in README


def test_readme_employment_to_age_feature_contract():
    assert "DAYS_EMPLOYED" in README or "source day values" in README
    assert "employed_years" in README and "age_years" in README
    assert "must not be claimed" in README or "must not be changed without retraining" in README
    assert "retraining" in README.lower()


def test_sql_and_preprocess_do_not_claim_unqualified_identity():
    assert "identical to app inference formula" not in SQL
    assert "exact raw-record identity" in SQL or "must not be changed without retraining" in PREPROCESS
    # Formula itself unchanged:
    assert "return round(float(employed_years) / age, 4)" in PREPROCESS


def test_app_batch_help_distinguishes_required_optional():
    assert "Missing required columns are rejected" in APP
    assert "Missing optional model inputs" in APP
    assert "Files with missing or misnamed columns will be rejected" not in APP
