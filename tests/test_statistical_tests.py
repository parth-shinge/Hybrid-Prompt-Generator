"""
tests/test_statistical_tests.py

Unit tests for Phase 4 statistical significance testing module.

Tests:
    - McNemar test with perfect / imperfect / identical predictions
    - Wilcoxon signed-rank with synthetic fold metrics
    - Bootstrap confidence intervals produce valid ranges
    - Bootstrap metric-difference CIs
    - run_statistical_tests orchestrator output structure
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from statistical_tests import (
    mcnemar_test,
    wilcoxon_fold_test,
    bootstrap_confidence_intervals,
    bootstrap_metric_difference,
    run_statistical_tests,
)


# ---- Fixtures ----

@pytest.fixture
def binary_predictions():
    """Synthetic binary test-set predictions for two models."""
    y_true  = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
    y_pred_good = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]  # perfect
    y_pred_bad  = ["b", "a", "b", "a", "a", "b", "a", "b", "a", "b"]  # two wrong
    y_prob_good = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    y_prob_bad  = np.array([0.8, 0.2, 0.7, 0.3, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    return y_true, y_pred_good, y_pred_bad, y_prob_good, y_prob_bad


@pytest.fixture
def fold_metrics_pair():
    """Simulated per-fold metrics for two models across 5 folds."""
    folds_a = [
        {"accuracy": 0.90, "f1": 0.88, "roc_auc": 0.92, "fold": 1},
        {"accuracy": 0.85, "f1": 0.83, "roc_auc": 0.89, "fold": 2},
        {"accuracy": 0.88, "f1": 0.86, "roc_auc": 0.91, "fold": 3},
        {"accuracy": 0.87, "f1": 0.85, "roc_auc": 0.90, "fold": 4},
        {"accuracy": 0.89, "f1": 0.87, "roc_auc": 0.93, "fold": 5},
    ]
    folds_b = [
        {"accuracy": 0.70, "f1": 0.68, "roc_auc": 0.75, "fold": 1},
        {"accuracy": 0.72, "f1": 0.70, "roc_auc": 0.77, "fold": 2},
        {"accuracy": 0.69, "f1": 0.67, "roc_auc": 0.74, "fold": 3},
        {"accuracy": 0.71, "f1": 0.69, "roc_auc": 0.76, "fold": 4},
        {"accuracy": 0.73, "f1": 0.71, "roc_auc": 0.78, "fold": 5},
    ]
    return folds_a, folds_b


# ---- McNemar Test ----

class TestMcNemarTest:
    def test_identical_predictions_pvalue_high(self, binary_predictions):
        y_true, y_pred_good, _, _, _ = binary_predictions
        result = mcnemar_test(y_true, y_pred_good, y_pred_good)
        # Both models identical → no discordant pairs → p should be 1.0
        assert result["p_value"] == 1.0
        assert result["significant_at_005"] is False

    def test_different_predictions_returns_structure(self, binary_predictions):
        y_true, y_pred_good, y_pred_bad, _, _ = binary_predictions
        result = mcnemar_test(y_true, y_pred_good, y_pred_bad)
        assert "contingency_table" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "significant_at_005" in result
        assert isinstance(result["contingency_table"], list)
        assert len(result["contingency_table"]) == 2

    def test_perfect_vs_imperfect(self, binary_predictions):
        y_true, y_pred_good, y_pred_bad, _, _ = binary_predictions
        result = mcnemar_test(y_true, y_pred_good, y_pred_bad)
        # Contingency entries should be non-negative integers
        for row in result["contingency_table"]:
            for val in row:
                assert val >= 0

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            mcnemar_test(["a", "b"], ["a"], ["b", "a"])


# ---- Wilcoxon Signed-Rank Test ----

class TestWilcoxonFoldTest:
    def test_significant_difference(self, fold_metrics_pair):
        folds_a, folds_b = fold_metrics_pair
        result = wilcoxon_fold_test(folds_a, folds_b)
        for metric in ["accuracy", "f1", "roc_auc"]:
            assert metric in result
            assert "statistic" in result[metric]
            assert "p_value" in result[metric]
            assert "n_folds" in result[metric]
            assert result[metric]["n_folds"] == 5

    def test_identical_folds_not_significant(self, fold_metrics_pair):
        folds_a, _ = fold_metrics_pair
        result = wilcoxon_fold_test(folds_a, folds_a)
        for metric in ["accuracy", "f1", "roc_auc"]:
            assert result[metric]["significant_at_005"] is False

    def test_few_folds_warning(self):
        """When < 5 folds are given, result should include a warning."""
        folds = [
            {"accuracy": 0.8, "f1": 0.7, "roc_auc": 0.9, "fold": 1},
            {"accuracy": 0.7, "f1": 0.6, "roc_auc": 0.8, "fold": 2},
        ]
        result = wilcoxon_fold_test(folds, folds, metric_keys=["accuracy"])
        assert "warning" in result["accuracy"]

    def test_custom_metric_keys(self, fold_metrics_pair):
        folds_a, folds_b = fold_metrics_pair
        result = wilcoxon_fold_test(folds_a, folds_b, metric_keys=["accuracy"])
        assert "accuracy" in result
        assert "f1" not in result


# ---- Bootstrap Confidence Intervals ----

class TestBootstrapCI:
    def test_ci_structure(self, binary_predictions):
        y_true, y_pred_good, _, y_prob_good, _ = binary_predictions
        result = bootstrap_confidence_intervals(
            y_true, y_pred_good, y_prob=y_prob_good,
            n_boot=100, seed=42,
        )
        for metric in ["accuracy", "f1", "roc_auc"]:
            assert metric in result
            ci = result[metric]
            assert "mean" in ci
            assert "ci_lower" in ci
            assert "ci_upper" in ci
            assert "std" in ci

    def test_ci_bounds_valid(self, binary_predictions):
        y_true, y_pred_good, _, y_prob_good, _ = binary_predictions
        result = bootstrap_confidence_intervals(
            y_true, y_pred_good, y_prob=y_prob_good,
            n_boot=200, seed=42,
        )
        for metric in ["accuracy", "f1"]:
            ci = result[metric]
            assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
            assert 0.0 <= ci["ci_lower"]
            assert ci["ci_upper"] <= 1.0

    def test_perfect_predictions_high_ci(self, binary_predictions):
        y_true, y_pred_good, _, y_prob_good, _ = binary_predictions
        result = bootstrap_confidence_intervals(
            y_true, y_pred_good, y_prob=y_prob_good,
            n_boot=200, seed=42,
        )
        # Perfect predictions should yield high CI lower bound
        assert result["accuracy"]["ci_lower"] >= 0.5

    def test_no_prob_skips_auc(self, binary_predictions):
        y_true, y_pred_good, _, _, _ = binary_predictions
        result = bootstrap_confidence_intervals(
            y_true, y_pred_good, y_prob=None,
            n_boot=50, seed=42,
        )
        # ROC-AUC should be NaN when no probabilities given
        assert np.isnan(result["roc_auc"]["mean"])


# ---- Bootstrap Metric Difference ----

class TestBootstrapDiff:
    def test_diff_structure(self, binary_predictions):
        y_true, y_pred_good, y_pred_bad, y_prob_good, y_prob_bad = binary_predictions
        result = bootstrap_metric_difference(
            y_true,
            y_pred_good, y_prob_good,
            y_pred_bad, y_prob_bad,
            n_boot=100, seed=42,
        )
        for metric in ["accuracy", "f1", "roc_auc"]:
            assert metric in result
            d = result[metric]
            assert "mean_diff" in d
            assert "ci_lower" in d
            assert "ci_upper" in d
            assert "ci_excludes_zero" in d

    def test_identical_models_zero_diff(self, binary_predictions):
        y_true, y_pred_good, _, y_prob_good, _ = binary_predictions
        result = bootstrap_metric_difference(
            y_true,
            y_pred_good, y_prob_good,
            y_pred_good, y_prob_good,
            n_boot=100, seed=42,
        )
        for metric in ["accuracy", "f1"]:
            d = result[metric]
            assert abs(d["mean_diff"]) < 1e-10
            assert d["ci_excludes_zero"] is False


# ---- Orchestrator ----

class TestRunStatisticalTests:
    def test_output_structure(self, binary_predictions, fold_metrics_pair):
        y_true, y_pred_good, y_pred_bad, y_prob_good, y_prob_bad = binary_predictions
        folds_a, folds_b = fold_metrics_pair

        cv_results = {
            "models": {
                "neural": {"fold_metrics": folds_a},
                "tfidf_lr": {"fold_metrics": folds_b},
                "embedding_lr": {"fold_metrics": folds_b},
            },
        }

        test_predictions = {
            "neural": {
                "y_true": y_true,
                "y_pred": y_pred_good,
                "y_prob": y_prob_good,
            },
            "tfidf_lr": {
                "y_true": y_true,
                "y_pred": y_pred_bad,
                "y_prob": y_prob_bad,
            },
            "embedding_lr": {
                "y_true": y_true,
                "y_pred": y_pred_bad,
                "y_prob": y_prob_bad,
            },
        }

        config = {
            "seed": 42,
            "statistical_tests": {
                "bootstrap_iterations": 50,
                "confidence_level": 0.95,
                "reference_model": "neural",
                "comparison_models": ["tfidf_lr", "embedding_lr"],
            },
        }

        result = run_statistical_tests(cv_results, test_predictions, config)

        assert "mcnemar" in result
        assert "wilcoxon" in result
        assert "bootstrap_ci" in result
        assert "bootstrap_diff_ci" in result
        assert "timestamp" in result
        assert "config" in result

        assert "neural_vs_tfidf_lr" in result["mcnemar"]
        assert "neural_vs_embedding_lr" in result["mcnemar"]
        assert "neural_vs_tfidf_lr" in result["wilcoxon"]
        assert "neural" in result["bootstrap_ci"]
        assert "neural_vs_tfidf_lr" in result["bootstrap_diff_ci"]

    def test_missing_model_handled(self):
        """Gracefully handles missing models in predictions."""
        cv_results = {"models": {}}
        test_predictions = {}
        config = {
            "statistical_tests": {
                "reference_model": "neural",
                "comparison_models": ["tfidf_lr"],
                "bootstrap_iterations": 10,
            },
        }

        result = run_statistical_tests(cv_results, test_predictions, config)

        assert result["mcnemar"]["neural_vs_tfidf_lr"].get("skipped") is True
        assert result["wilcoxon"]["neural_vs_tfidf_lr"].get("skipped") is True
