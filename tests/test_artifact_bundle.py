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

from model.artifact_bundle import (  # noqa: E402
    MANIFEST_HASHED_FILES,
    REQUIRED_BUNDLE_FILES,
    resolve_training_commit,
    write_serving_bundle,
)
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


def test_write_serving_bundle_creates_required_files_including_json_sidecar(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    manifest = write_serving_bundle(
        model, artifact, metadata, out, training_commit="testcommit", git_dirty=False
    )
    for name in REQUIRED_BUNDLE_FILES:
        assert (out / name).is_file()
    assert "preprocessing.json" in REQUIRED_BUNDLE_FILES
    assert "encoders.pkl" not in REQUIRED_BUNDLE_FILES
    assert not (out / "encoders.pkl").exists()

    booster_text = (out / "loaniq_booster.json").read_text(encoding="utf-8")
    assert json.loads(booster_text)
    sidecar = json.loads((out / "preprocessing.json").read_text(encoding="utf-8"))
    assert "medians" in sidecar
    assert "feature_order" in sidecar

    for key, record in manifest["artifacts"].items():
        path = out / Path(key).name
        assert path.stat().st_size == record["bytes"]
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]

    for name in MANIFEST_HASHED_FILES:
        assert f"model/{name}" in manifest["artifacts"]

    assert manifest["training_commit"] == "testcommit"
    assert manifest["git_dirty"] is False
    assert "inspection-only" in manifest["preprocessing_json_note"]
    assert "not a single multi-file filesystem transaction" in " ".join(
        manifest["notes"]
    ).lower() or "not a single multi-file" in " ".join(manifest["notes"])


def test_staging_dirs_removed_after_success(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    write_serving_bundle(model, artifact, metadata, out)
    leftovers = list(out.glob(".bundle_staging_*"))
    assert leftovers == []


def test_encoders_pkl_not_written_by_bundle(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    # Plant a prior encoders.pkl that must remain untouched by the bundle writer.
    prior = out / "encoders.pkl"
    out.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b"prior-encoders")
    write_serving_bundle(model, artifact, metadata, out)
    assert prior.read_bytes() == b"prior-encoders"


def test_failure_before_publish_leaves_previous_output_unchanged(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    write_serving_bundle(model, artifact, metadata, out)
    previous = {name: (out / name).read_bytes() for name in REQUIRED_BUNDLE_FILES}

    with mock.patch(
        "model.artifact_bundle.save_preprocessing",
        side_effect=RuntimeError("simulated staging failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated staging failure"):
            write_serving_bundle(model, artifact, metadata, out)

    for name, content in previous.items():
        assert (out / name).read_bytes() == content
    assert list(out.glob(".bundle_staging_*")) == []


def test_incomplete_publish_rolls_back_previous_bundle(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    write_serving_bundle(model, artifact, metadata, out)
    previous = {name: (out / name).read_bytes() for name in REQUIRED_BUNDLE_FILES}

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
    assert list(out.glob(".bundle_staging_*")) == []


def test_train_py_uses_shared_bundle_writer_without_independent_encoders_dump():
    text = (ROOT / "model" / "train.py").read_text(encoding="utf-8")
    assert "write_serving_bundle" in text
    assert "resolve_training_commit" in text
    assert 'joblib.dump(artifact["encoders"]' not in text
    assert 'joblib.dump(model, "model/loaniq_model.pkl")' not in text


def test_resolve_training_commit_explicit_and_env_and_git():
    assert resolve_training_commit("abc123")["training_commit"] == "abc123"
    assert (
        resolve_training_commit(None, env={"LOANIQ_TRAINING_COMMIT": "envsha"})[
            "training_commit"
        ]
        == "envsha"
    )
    # Explicit wins over env.
    assert (
        resolve_training_commit("cli", env={"LOANIQ_TRAINING_COMMIT": "envsha"})[
            "training_commit"
        ]
        == "cli"
    )


def test_resolve_training_commit_git_and_failure_null(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return mock.Mock(returncode=0, stdout="deadbeef\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return mock.Mock(returncode=0, stdout=" M file\n", stderr="")
        return mock.Mock(returncode=1, stdout="", stderr="err")

    monkeypatch.setattr("model.artifact_bundle.subprocess.run", fake_run)
    result = resolve_training_commit(None, env={}, repo_dir=tmp_path)
    assert result["training_commit"] == "deadbeef"
    assert result["git_dirty"] is True

    def boom(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr("model.artifact_bundle.subprocess.run", boom)
    result = resolve_training_commit(None, env={}, repo_dir=tmp_path)
    assert result["training_commit"] is None
    assert result["git_dirty"] is None


def test_provenance_failure_does_not_break_bundle(tmp_path):
    model, artifact, metadata = _tiny_xgb_model()
    out = tmp_path / "bundle"
    manifest = write_serving_bundle(
        model, artifact, metadata, out, training_commit=None, git_dirty=None
    )
    assert manifest["training_commit"] is None
    assert (out / "loaniq_booster.json").is_file()


def test_committed_production_artifacts_untouched_by_bundle_tests():
    # Sanity: production paths still present with expected names (hashes checked in acceptance).
    for name in (
        "loaniq_model.pkl",
        "preprocessing.pkl",
        "loaniq_booster.json",
        "metadata.json",
        "artifact_manifest.json",
    ):
        assert (ROOT / "model" / name).is_file()
