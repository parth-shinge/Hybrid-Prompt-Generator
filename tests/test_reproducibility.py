"""
tests/test_reproducibility.py

Tests for Phase 1 reproducibility utilities:
- SHA-256 hashing (file and in-memory)
- Git commit detection
- ExperimentTracker metadata persistence
- Deterministic seeding
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- Hashing ----

class TestHashing:
    def test_compute_file_sha256_consistent(self, tmp_path):
        """Same file content always produces the same hash."""
        from utils.hashing import compute_file_sha256

        fp = tmp_path / "test.csv"
        fp.write_text("col1,col2\na,1\nb,2\n", encoding="utf-8")
        h1 = compute_file_sha256(fp)
        h2 = compute_file_sha256(fp)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_compute_file_sha256_different_content(self, tmp_path):
        from utils.hashing import compute_file_sha256

        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")
        assert compute_file_sha256(f1) != compute_file_sha256(f2)

    def test_compute_file_sha256_missing_file(self):
        from utils.hashing import compute_file_sha256

        with pytest.raises(FileNotFoundError):
            compute_file_sha256("/nonexistent/file.csv")

    def test_compute_data_sha256_deterministic(self):
        from utils.hashing import compute_data_sha256

        data = [("text1", "label1"), ("text2", "label2")]
        h1 = compute_data_sha256(data)
        h2 = compute_data_sha256(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_data_sha256_order_sensitive(self):
        from utils.hashing import compute_data_sha256

        d1 = [("a", "1"), ("b", "2")]
        d2 = [("b", "2"), ("a", "1")]
        assert compute_data_sha256(d1) != compute_data_sha256(d2)


# ---- Git info ----

class TestGitInfo:
    def test_get_git_commit_returns_string(self):
        from utils.git_info import get_git_commit

        result = get_git_commit()
        assert isinstance(result, str)
        # Either a 40-char hex hash or "unknown"
        assert len(result) == 40 or result == "unknown"

    def test_is_dirty_returns_bool_or_none(self):
        from utils.git_info import is_dirty

        result = is_dirty()
        assert result is None or isinstance(result, bool)


# ---- Experiment tracker ----

class TestExperimentTracker:
    def test_save_metadata_creates_valid_json(self, tmp_path):
        from utils.experiment import ExperimentTracker

        config = {"seed": 42, "neural_ranker": {"lr": 0.001}}
        tracker = ExperimentTracker(config, seed=42)
        tracker.set_dataset_hash("abc123def456")

        exp_dir = tracker.save_metadata(base_dir=str(tmp_path))

        meta_path = exp_dir / "metadata.json"
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # All required keys present
        required = {"experiment_id", "timestamp", "seed", "dataset_hash",
                     "git_commit", "config"}
        assert required.issubset(set(meta.keys()))
        assert meta["seed"] == 42
        assert meta["dataset_hash"] == "abc123def456"

    def test_save_metadata_creates_config_snapshot(self, tmp_path):
        from utils.experiment import ExperimentTracker

        config = {"seed": 42, "ranker": {"type": "neural"}}
        tracker = ExperimentTracker(config, seed=42)
        exp_dir = tracker.save_metadata(base_dir=str(tmp_path))

        cfg_path = exp_dir / "config_snapshot.yaml"
        assert cfg_path.exists()

    def test_save_metrics(self, tmp_path):
        from utils.experiment import ExperimentTracker

        tracker = ExperimentTracker({"seed": 42}, seed=42)
        tracker.save_metadata(base_dir=str(tmp_path))
        metrics_path = tracker.save_metrics(
            {"loss": 0.12, "acc": 0.93},
            base_dir=str(tmp_path),
        )
        assert metrics_path.exists()
        with open(metrics_path, "r") as f:
            m = json.load(f)
        assert m["acc"] == 0.93


# ---- Seeding ----

class TestSeeding:
    def test_numpy_determinism(self):
        from seeds import set_deterministic

        set_deterministic(123)
        a = np.random.rand(5)
        set_deterministic(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        from seeds import set_deterministic

        set_deterministic(1)
        a = np.random.rand(5)
        set_deterministic(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)
