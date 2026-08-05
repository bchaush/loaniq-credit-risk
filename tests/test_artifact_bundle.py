"""Serving-bundle writer tests (no production retrain / no committed artifact writes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.artifact_bundle import REQUIRED_BUNDLE_FILES, write_serving_bundle  # noqa: E402
from model.preprocess import fit_preprocessing  # noqa: E402


def _tiny_xgb_model():
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier

    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = np.array([0, 1, 0, 1])
    model = XGBClassifier(
        n_estimators=2,
        max_depth=1,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X, y)
    model._loaniq_best_iteration = 1
    model._loaniq_n_trees_served = 2
    artifact = fit_preprocessing(X, feature_order=list(X.columns))
    metadata = {
        "features": list(X.columns),
        "n_features": 2,
        "best_iteration": 1,
        "n_trees_served": 2,
        "preprocessing_version": artifact["preprocessing_version"],
        "roc_auc": 0.5,
        "pr_auc": 0.5,
        "n_train": 4,
        "n_val": 0,
        "n_test": 0,
        "default_rate": 0.5,
        "top_features": ["a", "b"],
        "decision_thresholds": {"approved_lt": 0.15, "review_lt": 0.35},
        "early_stopping_eval_set": "validation",
    }
    return model, artifact, metadata


def test_write_serving_bundle_creates_required_files(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    manifest = write_serving_bundle(
        model, artifact, metadata, out, training_commit="testcommit"
    )
    for name in REQUIRED_BUNDLE_FILES:
        assert (out / name).is_file()
    # Native booster via supported API — JSON parses and is non-empty.
    booster_text = (out / "loaniq_booster.json").read_text(encoding="utf-8")
    assert json.loads(booster_text)
    loaded = joblib.load(out / "loaniq_model.pkl")
    assert hasattr(loaded, "get_booster")

    for key, record in manifest["artifacts"].items():
        path = out / Path(key).name
        assert path.stat().st_size == record["bytes"]
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]

    assert manifest["n_features"] == 2
    assert manifest["best_iteration"] == 1
    assert manifest["n_trees_served"] == 2
    assert manifest["inference_path"] == "model/loaniq_booster.json"
    assert manifest["pickle_reference"] == "model/loaniq_model.pkl"
    assert manifest["training_commit"] == "testcommit"
    assert "python_version" in manifest
    assert "numpy_version" in manifest
    assert "pandas_version" in manifest
    assert "joblib_version" in manifest
    assert "scikit_learn_version" in manifest
    assert "xgboost_version" in manifest


def test_incomplete_publish_does_not_clobber_previous_bundle(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    write_serving_bundle(model, artifact, metadata, out)
    previous = {
        name: (out / name).read_bytes() for name in REQUIRED_BUNDLE_FILES if name != "artifact_manifest.json"
    }
    # Also keep manifest bytes
    previous["artifact_manifest.json"] = (out / "artifact_manifest.json").read_bytes()

    calls = {"n": 0}
    real_replace = __import__("os").replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated publish failure")
        return real_replace(src, dst)

    with mock.patch("model.artifact_bundle.os.replace", side_effect=flaky_replace):
        with pytest.raises(OSError, match="simulated publish failure"):
            write_serving_bundle(model, artifact, metadata, out)

    for name, content in previous.items():
        assert (out / name).read_bytes() == content


def test_train_py_uses_shared_bundle_writer():
    text = (ROOT / "model" / "train.py").read_text(encoding="utf-8")
    assert "write_serving_bundle" in text
    assert "from artifact_bundle import write_serving_bundle" in text
    # Must not dump the pkl alone without the shared bundle path.
    assert "joblib.dump(model, \"model/loaniq_model.pkl\")" not in text
