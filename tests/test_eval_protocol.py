"""
tests/test_eval_protocol.py

Tests for Phase 3 evaluation protocol:
- Baselines (Random, Popularity, TF-IDF+LR)
- Metric computation
- Stratified CV engine
- Held-out test evaluation
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baselines import RandomBaseline, PopularityBaseline, TFIDFBaseline
from eval_protocol import compute_metrics, run_stratified_cv, run_held_out_test


# ---- Fixtures ----

@pytest.fixture
def sample_data():
    """Small synthetic dataset for testing."""
    texts = [
        "design a modern logo graphic",
        "create marketing content blog",
        "write persuasive ad copy for social media",
        "build a brand identity visual",
        "generate an engaging headline for blog",
        "develop a color palette for website",
        "draft email newsletter for campaign",
        "plan social media strategy content",
        "create infographic data visualization",
        "write product description for ecommerce",
        "design a poster for event promotion",
        "build a responsive landing page layout",
    ]
    labels = [
        "offline", "gemini", "offline", "gemini",
        "offline", "gemini", "offline", "gemini",
        "offline", "gemini", "offline", "gemini",
    ]
    return texts, labels


# ---- Baselines ----

class TestRandomBaseline:
    def test_fit_predict(self, sample_data):
        texts, labels = sample_data
        model = RandomBaseline(seed=42)
        model.fit(texts, labels)
        preds = model.predict(texts)
        assert len(preds) == len(texts)
        assert all(p in ("offline", "gemini") for p in preds)

    def test_predict_proba_shape(self, sample_data):
        texts, labels = sample_data
        model = RandomBaseline(seed=42)
        model.fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_determinism(self, sample_data):
        texts, labels = sample_data
        m1 = RandomBaseline(seed=7).fit(texts, labels)
        m2 = RandomBaseline(seed=7).fit(texts, labels)
        assert m1.predict(texts) == m2.predict(texts)


class TestPopularityBaseline:
    def test_predicts_majority(self, sample_data):
        texts, labels = sample_data
        # Balanced, so majority is first alphabetically with ties
        model = PopularityBaseline().fit(texts, labels)
        preds = model.predict(texts)
        assert all(p == model.majority_label_ for p in preds)

    def test_predict_proba_shape(self, sample_data):
        texts, labels = sample_data
        model = PopularityBaseline().fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_imbalanced(self):
        texts = ["a", "b", "c", "d", "e"]
        labels = ["x", "x", "x", "y", "y"]
        model = PopularityBaseline().fit(texts, labels)
        assert model.majority_label_ == "x"
        preds = model.predict(texts)
        assert all(p == "x" for p in preds)


class TestTFIDFBaseline:
    def test_fit_predict(self, sample_data):
        texts, labels = sample_data
        model = TFIDFBaseline(seed=42)
        model.fit(texts, labels)
        preds = model.predict(texts)
        assert len(preds) == len(texts)
        assert all(p in ("offline", "gemini") for p in preds)

    def test_predict_proba_shape(self, sample_data):
        texts, labels = sample_data
        model = TFIDFBaseline(seed=42)
        model.fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_from_config(self, sample_data):
        texts, labels = sample_data
        config = {
            "seed": 42,
            "tfidf_ranker": {
                "max_features": 1000,
                "ngram_range": [1, 1],
                "solver": "liblinear",
                "max_iter": 500,
            },
        }
        model = TFIDFBaseline.from_config(config)
        model.fit(texts, labels)
        preds = model.predict(texts)
        assert len(preds) == len(texts)


# ---- Metrics ----

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "a", "b"]
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        m = compute_metrics(y_true, y_pred, y_prob=y_prob, pos_label="b")
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_worst_predictions(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["b", "b", "a", "a"]
        m = compute_metrics(y_true, y_pred, pos_label="b")
        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_required_keys(self):
        m = compute_metrics(["a", "b"], ["a", "b"])
        assert set(m.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    def test_all_values_numeric(self):
        m = compute_metrics(["a", "b", "a"], ["a", "a", "b"])
        for v in m.values():
            assert isinstance(v, float)


# ---- CV Engine ----

class TestStratifiedCV:
    def test_runs_without_error(self, sample_data):
        texts, labels = sample_data
        texts_arr = np.array(texts)
        labels_arr = np.array(labels)
        result = run_stratified_cv(
            lambda: RandomBaseline(seed=42),
            texts_arr, labels_arr,
            n_folds=3, seed=42,
        )
        assert "fold_metrics" in result
        assert "mean_metrics" in result
        assert "std_metrics" in result
        assert len(result["fold_metrics"]) == 3

    def test_fold_metrics_have_required_keys(self, sample_data):
        texts, labels = sample_data
        result = run_stratified_cv(
            lambda: PopularityBaseline(),
            np.array(texts), np.array(labels),
            n_folds=2, seed=42,
        )
        for fm in result["fold_metrics"]:
            assert "accuracy" in fm
            assert "precision" in fm
            assert "recall" in fm
            assert "f1" in fm
            assert "fold" in fm

    def test_tfidf_baseline_cv(self, sample_data):
        texts, labels = sample_data
        result = run_stratified_cv(
            lambda: TFIDFBaseline(seed=42),
            np.array(texts), np.array(labels),
            n_folds=2, seed=42,
        )
        m = result["mean_metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0


# ---- Held-out test ----

class TestHeldOutTest:
    def test_runs_without_error(self, sample_data):
        texts, labels = sample_data
        n = len(texts)
        split = n // 2
        result = run_held_out_test(
            lambda: RandomBaseline(seed=42),
            texts[:split], labels[:split],
            texts[split:], labels[split:],
            seed=42,
        )
        assert "accuracy" in result
        assert "f1" in result
        assert 0.0 <= result["accuracy"] <= 1.0
