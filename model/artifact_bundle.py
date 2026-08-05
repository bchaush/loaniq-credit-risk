"""Serving-bundle writer for future training runs.

Does not retrain or mutate committed production artifacts unless explicitly
pointed at an output directory by the caller (tests use temp dirs).

Publication model: files are written into a staging directory, hashed and
validated, then published with per-file backups and rollback on failure.
Each individual ``os.replace`` is an atomic filesystem operation; the complete
multi-file bundle is not claimed to be one indivisible transaction.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

try:
    from model.preprocess import save_preprocessing
except ImportError:  # train.py adds model/ to sys.path
    from preprocess import save_preprocessing  # type: ignore

# Runtime-critical + inspection sidecar published together.
REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "loaniq_model.pkl",
    "preprocessing.pkl",
    "preprocessing.json",  # inspection-only sidecar; runtime loads .pkl
    "loaniq_booster.json",
    "metadata.json",
    "artifact_manifest.json",
)

# Hashed in the generated manifest (manifest file itself is not self-hashed).
MANIFEST_HASHED_FILES: tuple[str, ...] = (
    "loaniq_model.pkl",
    "preprocessing.pkl",
    "preprocessing.json",
    "loaniq_booster.json",
    "metadata.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}."
        f"{os.sys.version_info.micro}",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
    }


def resolve_training_commit(
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve training provenance without inventing values.

    Priority:
    1. Explicit argument
    2. ``LOANIQ_TRAINING_COMMIT`` environment variable
    3. ``git rev-parse HEAD`` when available

    Returns ``{"training_commit": str|None, "git_dirty": bool|None}``.
    Lookup failures yield null commit / null dirty — they do not raise.
    """
    environ = env if env is not None else os.environ
    commit: str | None = None
    if explicit is not None and str(explicit).strip():
        commit = str(explicit).strip()
    else:
        env_val = environ.get("LOANIQ_TRAINING_COMMIT")
        if env_val is not None and str(env_val).strip():
            commit = str(env_val).strip()

    git_dirty: bool | None = None
    cwd = str(repo_dir) if repo_dir is not None else None

    if commit is None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                commit = proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = None

    try:
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if dirty_proc.returncode == 0:
            git_dirty = bool(dirty_proc.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        git_dirty = None

    return {"training_commit": commit, "git_dirty": git_dirty}


def write_serving_bundle(
    trained_model: Any,
    preprocessing_artifact: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    training_commit: str | None = None,
    creation_date: str | None = None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    """Write a complete serving bundle into *output_dir* with rollback protection.

    Staging writes happen in a temporary sibling directory. Existing outputs are
    replaced only after every required artifact is present and hashed. On
    mid-publish failure, previously valid files are restored from backups.

    Does not write ``encoders.pkl`` (legacy; superseded by ``preprocessing.pkl``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_root = output_dir / f".bundle_staging_{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    backups: dict[str, Path] = {}
    published: list[str] = []

    try:
        model_path = staging_root / "loaniq_model.pkl"
        prep_path = staging_root / "preprocessing.pkl"
        booster_path = staging_root / "loaniq_booster.json"
        meta_path = staging_root / "metadata.json"
        manifest_path = staging_root / "artifact_manifest.json"

        joblib.dump(trained_model, model_path)
        # Writes preprocessing.pkl and preprocessing.json into staging only.
        save_preprocessing(preprocessing_artifact, prep_path)

        booster = trained_model.get_booster()
        booster.save_model(str(booster_path))

        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        created = creation_date or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        versions = _package_versions()
        best_iteration = int(
            metadata.get(
                "best_iteration", getattr(trained_model, "_loaniq_best_iteration", 0)
            )
        )
        n_trees = int(
            metadata.get(
                "n_trees_served",
                getattr(trained_model, "_loaniq_n_trees_served", best_iteration + 1),
            )
        )
        n_features = int(metadata.get("n_features", len(metadata.get("features", []))))
        prep_version = str(
            metadata.get(
                "preprocessing_version",
                preprocessing_artifact.get("preprocessing_version", "1.0"),
            )
        )

        artifact_records: dict[str, dict[str, Any]] = {}
        for name in MANIFEST_HASHED_FILES:
            path = staging_root / name
            if not path.is_file():
                raise RuntimeError(f"Missing staged artifact: {name}")
            artifact_records[f"model/{name}"] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }

        manifest: dict[str, Any] = {
            "created_at_utc": created,
            "training_commit": training_commit,
            "git_dirty": git_dirty,
            **versions,
            "preprocessing_schema_version": prep_version,
            "n_features": n_features,
            "best_iteration": best_iteration,
            "n_trees_served": n_trees,
            "inference_path": "model/loaniq_booster.json",
            "pickle_reference": "model/loaniq_model.pkl",
            "preprocessing_reference": "model/preprocessing.pkl",
            "preprocessing_json_note": (
                "preprocessing.json is an inspection-only sidecar; "
                "runtime loads preprocessing.pkl."
            ),
            "notes": [
                "Native XGBoost JSON booster is the runtime inference artifact.",
                "Bundle written via staging with rollback-protected publication "
                "(per-file os.replace; not a single multi-file filesystem transaction).",
                "encoders.pkl is not part of the serving bundle; categorical maps "
                "live inside preprocessing.pkl.",
            ],
            "artifacts": artifact_records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        for name in REQUIRED_BUNDLE_FILES:
            if not (staging_root / name).is_file():
                raise RuntimeError(f"Incomplete staging bundle; missing {name}")

        for name in REQUIRED_BUNDLE_FILES:
            dest = output_dir / name
            staged = staging_root / name
            if dest.exists():
                bak = staging_root / f"{name}.bak"
                shutil.copy2(dest, bak)
                backups[name] = bak
            os.replace(staged, dest)
            published.append(name)

        return manifest
    except Exception:
        for name in published:
            dest = output_dir / name
            bak = backups.get(name)
            if bak is not None and bak.exists():
                os.replace(bak, dest)
            elif dest.exists() and name in published:
                try:
                    dest.unlink()
                except OSError:
                    pass
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
