"""
tests/test_shap_explain.py

Unit tests for Phase 6 — SHAP Interpretability.

Tests the core functions of shap_explain.py using mock models
and embeddings, without requiring a trained checkpoint or
sentence-transformers.
"""

import json
import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- Helpers for mocking ----

class _MockModel:
    """A minimal mock that behaves like PromptRankerNet for testing."""

    def eval(self):
        return self

    def __call__(self, x):
        import torch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        out = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        return out

    def parameters(self):
        return iter([])


def _mock_load_model_and_embed(texts, model_path="", config_path=""):
    """Mock replacement for shap_explain._load_model_and_embed."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(texts), 16).astype(np.float32)
    label_map = {"gemini": 0, "offline": 1}
    config = {
        "shap": {
            "background_samples": 5,
            "max_display_dims": 10,
            "model_path": "mock.pt",
        },
        "neural_ranker": {"embed_model": "mock", "embed_dim": 16},
    }
    return _MockModel(), embeddings, label_map, config


# ---- Fixtures ----

@pytest.fixture
def patch_shap_module():
    """Patch the internal helper in shap_explain to avoid loading the real model."""
    with mock.patch(
        "shap_explain._load_model_and_embed",
        side_effect=_mock_load_model_and_embed,
    ):
        yield


@pytest.fixture
def patch_local_shap():
    """Patch neural_ranker imports used inside compute_local_shap."""
    mock_model = _MockModel()
    rng = np.random.RandomState(42)

    def _mock_load_checkpoint(path):
        return mock_model, {"gemini": 0, "offline": 1}

    def _mock_embed_texts(texts, model_name=""):
        return rng.randn(len(texts), 16).astype(np.float32)

    def _mock_load_config(path=""):
        return {
            "shap": {"background_samples": 5, "max_display_dims": 10, "model_path": "mock.pt"},
            "neural_ranker": {"embed_model": "mock", "embed_dim": 16},
        }

    def _mock_neural_cfg(config):
        return config.get("neural_ranker", {"embed_model": "mock", "embed_dim": 16})

    # Patch at the import location inside shap_explain.compute_local_shap
    with mock.patch.dict("sys.modules", {
        "neural_ranker": mock.MagicMock(
            _load_checkpoint=_mock_load_checkpoint,
            _embed_texts=_mock_embed_texts,
            load_config=_mock_load_config,
            _neural_cfg=_mock_neural_cfg,
        ),
    }):
        yield


@pytest.fixture
def tmp_results_dir():
    """Create a temp directory for results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---- Tests: _shap_config ----

class TestShapConfig:
    def test_defaults(self):
        from shap_explain import _shap_config
        cfg = _shap_config({})
        assert cfg["background_samples"] == 50
        assert cfg["max_display_dims"] == 20
        assert cfg["model_path"] == "models/neural_ranker_best.pt"

    def test_override(self):
        from shap_explain import _shap_config
        cfg = _shap_config({"shap": {"background_samples": 100}})
        assert cfg["background_samples"] == 100
        assert cfg["max_display_dims"] == 20


# ---- Tests: plot functions ----

class TestPlotGlobalImportance:
    def test_empty_result(self):
        from shap_explain import plot_global_importance
        fig = plot_global_importance({"top_dims": []})
        assert fig is not None

    def test_with_data(self):
        from shap_explain import plot_global_importance
        import matplotlib.pyplot as plt
        result = {
            "top_dims": [
                {"dim": 0, "label": "dim_0", "importance": 0.5},
                {"dim": 1, "label": "dim_1", "importance": 0.3},
                {"dim": 2, "label": "dim_2", "importance": 0.1},
            ],
        }
        fig = plot_global_importance(result, max_dims=3)
        assert fig is not None
        plt.close(fig)


class TestPlotLocalExplanation:
    def test_empty_result(self):
        from shap_explain import plot_local_explanation
        fig = plot_local_explanation({"top_contributions": []})
        assert fig is not None

    def test_with_data(self):
        from shap_explain import plot_local_explanation
        import matplotlib.pyplot as plt
        pred = {
            "predicted_label": "offline",
            "probability": 0.75,
            "top_contributions": [
                {"dim": 0, "label": "dim_0", "shap_value": 0.2, "abs_value": 0.2},
                {"dim": 1, "label": "dim_1", "shap_value": -0.1, "abs_value": 0.1},
            ],
        }
        fig = plot_local_explanation(pred, max_dims=5)
        assert fig is not None
        plt.close(fig)


# ---- Tests: compute_global_shap ----

class TestComputeGlobalShap:
    def test_basic_run(self, patch_shap_module, tmp_results_dir):
        from shap_explain import compute_global_shap

        texts = [f"Test prompt number {i}" for i in range(20)]
        result = compute_global_shap(
            texts, model_path="mock.pt", config_path="mock.yaml",
            max_samples=15, results_dir=tmp_results_dir,
        )

        assert "mean_abs_shap" in result
        assert "top_dims" in result
        assert "n_samples" in result
        assert result["n_samples"] <= 15
        assert len(result["top_dims"]) > 0
        assert "timestamp" in result
        assert os.path.exists(os.path.join(tmp_results_dir, "shap_global.json"))

    def test_result_structure(self, patch_shap_module, tmp_results_dir):
        from shap_explain import compute_global_shap

        texts = [f"Prompt {i}" for i in range(10)]
        result = compute_global_shap(
            texts, model_path="mock.pt", config_path="mock.yaml",
            results_dir=tmp_results_dir,
        )

        for entry in result["top_dims"]:
            assert "dim" in entry
            assert "label" in entry
            assert "importance" in entry
            assert isinstance(entry["importance"], float)

    def test_saved_json_valid(self, patch_shap_module, tmp_results_dir):
        from shap_explain import compute_global_shap

        texts = [f"Sample {i}" for i in range(10)]
        compute_global_shap(
            texts, model_path="mock.pt", config_path="mock.yaml",
            results_dir=tmp_results_dir,
        )

        path = os.path.join(tmp_results_dir, "shap_global.json")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "mean_abs_shap" in data
        assert isinstance(data["mean_abs_shap"], list)


# ---- Tests: compute_local_shap ----

class TestComputeLocalShap:
    def test_single_text(self, patch_local_shap, tmp_results_dir):
        from shap_explain import compute_local_shap

        result = compute_local_shap(
            "Explain this single prompt about AI.",
            model_path="mock.pt", config_path="mock.yaml",
            background_texts=None, results_dir=tmp_results_dir,
        )

        assert "predictions" in result
        assert len(result["predictions"]) == 1
        pred = result["predictions"][0]
        assert "predicted_label" in pred
        assert "probability" in pred
        assert "top_contributions" in pred
        assert pred["predicted_label"] in ("gemini", "offline", "unknown")

    def test_multiple_texts(self, patch_local_shap, tmp_results_dir):
        from shap_explain import compute_local_shap

        result = compute_local_shap(
            ["Prompt one.", "Prompt two."],
            model_path="mock.pt", config_path="mock.yaml",
            results_dir=tmp_results_dir,
        )
        assert len(result["predictions"]) == 2

    def test_with_background_texts(self, patch_local_shap, tmp_results_dir):
        from shap_explain import compute_local_shap

        bg = [f"Background {i}" for i in range(10)]
        result = compute_local_shap(
            "Target prompt.",
            model_path="mock.pt", config_path="mock.yaml",
            background_texts=bg, results_dir=tmp_results_dir,
        )
        assert result["n_background"] == 10

    def test_saved_json_valid(self, patch_local_shap, tmp_results_dir):
        from shap_explain import compute_local_shap

        compute_local_shap(
            "A test prompt.",
            model_path="mock.pt", config_path="mock.yaml",
            results_dir=tmp_results_dir,
        )

        path = os.path.join(tmp_results_dir, "shap_local.json")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "predictions" in data


# ---- Tests: _ensure_results_dir ----

class TestEnsureResultsDir:
    def test_creates_directory(self):
        from shap_explain import _ensure_results_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "nested", "results")
            p = _ensure_results_dir(target)
            assert p.exists()
            assert p.is_dir()
