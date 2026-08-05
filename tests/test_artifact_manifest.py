"""Artifact manifest enforcement tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_artifact_compatibility import (  # noqa: E402
    load_manifest,
    sha256,
    verify_manifest_artifacts,
)


def test_real_manifest_passes():
    manifest = load_manifest()
    verify_manifest_artifacts(manifest, root=ROOT)


def test_modified_sha_fails(tmp_path: Path):
    manifest = load_manifest()
    # Point one artifact at a temp copy and corrupt the expected hash.
    src = ROOT / "model" / "metadata.json"
    dst = tmp_path / "metadata.json"
    dst.write_bytes(src.read_bytes())
    bad = {
        **manifest,
        "artifacts": {
            "metadata.json": {
                "sha256": "0" * 64,
                "bytes": dst.stat().st_size,
            }
        },
    }
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        verify_manifest_artifacts(bad, root=tmp_path)


def test_modified_byte_count_fails(tmp_path: Path):
    src = ROOT / "model" / "metadata.json"
    dst = tmp_path / "metadata.json"
    dst.write_bytes(src.read_bytes())
    bad = {
        "artifacts": {
            "metadata.json": {
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size + 1,
            }
        }
    }
    with pytest.raises(AssertionError, match="byte-size mismatch"):
        verify_manifest_artifacts(bad, root=tmp_path)


def test_missing_file_fails(tmp_path: Path):
    bad = {
        "artifacts": {
            "missing.bin": {"sha256": "abc", "bytes": 1},
        }
    }
    with pytest.raises(AssertionError, match="missing"):
        verify_manifest_artifacts(bad, root=tmp_path)


def test_invalid_required_field_fails(tmp_path: Path):
    manifest = load_manifest()
    broken = dict(manifest)
    broken.pop("n_features")
    path = tmp_path / "artifact_manifest.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(AssertionError, match="missing required field"):
        load_manifest(path)
