"""
statistical_tests.py

Phase 4 — Statistical Significance Testing for the Hybrid Prompt Generator.

Implements:
    1. McNemar test        — paired comparison of two classifiers on the same test set
    2. Wilcoxon signed-rank — paired comparison of per-fold CV metrics
    3. Bootstrap CIs       — non-parametric 95 % confidence intervals for metrics

Compatible with:
    - cv_results.json  (fold-level metrics for Wilcoxon)
    - test_results.json (aggregated held-out metrics)
    - config.yaml      (bootstrap_iterations, confidence_level, model names)
    - utils.experiment.ExperimentTracker

Usage:
    # Standalone
    python statistical_tests.py --config config.yaml

    # Programmatic (called from eval_protocol.py)
    from statistical_tests import run_statistical_tests
    stats = run_statistical_tests(cv_results, test_predictions, config)
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seeds import set_deterministic


# ============================================================
# 1.  McNemar Test
# ============================================================

def mcnemar_test(
    y_true: List[str],
    y_pred_a: List[str],
    y_pred_b: List[str],
) -> Dict[str, Any]:
    """McNemar test for paired nominal data.

    Compares two classifiers on the same test set by examining the
    2×2 contingency table of correct/incorrect classifications.

    Args:
        y_true:   Ground-truth labels.
        y_pred_a: Predictions from model A (reference, e.g. neural).
        y_pred_b: Predictions from model B (baseline).

    Returns:
        Dict with keys: contingency_table, statistic, p_value, significant_at_005.
    """
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    n = len(y_true)
    assert len(y_pred_a) == n and len(y_pred_b) == n, (
        "All prediction lists must have the same length"
    )

    correct_a = np.array([p == t for p, t in zip(y_pred_a, y_true)])
    correct_b = np.array([p == t for p, t in zip(y_pred_b, y_true)])

    # Contingency table:
    #              B correct   B wrong
    # A correct  [ n00,        n01   ]
    # A wrong    [ n10,        n11   ]
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))

    table = np.array([[n00, n01], [n10, n11]])

    # Use exact binomial test when discordant counts are small (≤25),
    # otherwise use chi-square approximation with continuity correction.
    exact = (n01 + n10) <= 25
    result = sm_mcnemar(table, exact=exact, correction=True)

    return {
        "contingency_table": table.tolist(),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "exact_test": exact,
        "significant_at_005": bool(result.pvalue < 0.05),
    }


# ============================================================
# 2.  Wilcoxon Signed-Rank Test
# ============================================================

def wilcoxon_fold_test(
    fold_metrics_a: List[Dict[str, float]],
    fold_metrics_b: List[Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Wilcoxon signed-rank test on per-fold CV metrics.

    Compares paired fold-level metric values between two models –
    appropriate when folds are the blocking variable.

    Args:
        fold_metrics_a: Per-fold metric dicts from model A (reference).
        fold_metrics_b: Per-fold metric dicts from model B (comparison).
        metric_keys:    Metrics to compare (default: accuracy, f1, roc_auc).

    Returns:
        Dict keyed by metric → {statistic, p_value, significant_at_005, n_folds}.
    """
    from scipy.stats import wilcoxon

    if metric_keys is None:
        metric_keys = ["accuracy", "f1", "roc_auc"]

    n = len(fold_metrics_a)
    assert n == len(fold_metrics_b), "Fold lists must have the same length"

    results: Dict[str, Dict[str, Any]] = {}

    for key in metric_keys:
        vals_a = np.array([fm[key] for fm in fold_metrics_a])
        vals_b = np.array([fm[key] for fm in fold_metrics_b])

        # Filter out NaN pairs
        valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
        va, vb = vals_a[valid], vals_b[valid]

        if len(va) < 5:
            results[key] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "significant_at_005": False,
                "n_folds": int(len(va)),
                "warning": (
                    f"Only {len(va)} valid folds — Wilcoxon requires ≥5 "
                    "paired observations for meaningful results."
                ),
            }
            continue

        diff = va - vb
        if np.allclose(diff, 0):
            results[key] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant_at_005": False,
                "n_folds": int(len(va)),
                "note": "All differences are zero.",
            }
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval = wilcoxon(va, vb, alternative="two-sided")

        results[key] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "significant_at_005": bool(pval < 0.05),
            "n_folds": int(len(va)),
        }

    return results


# ============================================================
# 3.  Bootstrap Confidence Intervals
# ============================================================

def _compute_metric(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_prob: Optional[np.ndarray],
    metric: str,
) -> float:
    """Compute a single metric value from binary arrays."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if metric == "accuracy":
        return float(accuracy_score(y_true_bin, y_pred_bin))
    elif metric == "f1":
        return float(f1_score(y_true_bin, y_pred_bin, zero_division=0.0))
    elif metric == "roc_auc":
        if y_prob is None:
            return float("nan")
        try:
            return float(roc_auc_score(y_true_bin, y_prob))
        except ValueError:
            return float("nan")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def bootstrap_confidence_intervals(
    y_true: List[str],
    y_pred: List[str],
    y_prob: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    pos_label: Optional[str] = None,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Non-parametric bootstrap 95 % confidence intervals.

    Resamples the held-out test predictions with replacement and
    computes the CI for each metric.

    Args:
        y_true:           Ground-truth label strings.
        y_pred:           Predicted label strings.
        y_prob:           Predicted probability for the positive class.
        n_boot:           Number of bootstrap iterations.
        confidence_level: Confidence level for the CI (default 0.95).
        seed:             Random seed.
        pos_label:        Label treated as positive.
        metric_keys:      Metrics to compute CIs for (default: accuracy, f1, roc_auc).

    Returns:
        Dict keyed by metric → {mean, ci_lower, ci_upper, std}.
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "f1", "roc_auc"]

    classes = sorted(set(y_true))
    if pos_label is None:
        pos_label = classes[-1] if len(classes) == 2 else classes[0]

    y_true_bin = np.array([1 if y == pos_label else 0 for y in y_true])
    y_pred_bin = np.array([1 if y == pos_label else 0 for y in y_pred])

    alpha = 1.0 - confidence_level
    rng = np.random.RandomState(seed)
    n = len(y_true_bin)

    boot_scores: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true_bin[idx]
        yp = y_pred_bin[idx]
        ypr = y_prob[idx] if y_prob is not None else None

        # Skip degenerate samples where all labels are the same class
        if len(np.unique(yt)) < 2:
            continue

        for key in metric_keys:
            val = _compute_metric(yt, yp, ypr, key)
            boot_scores[key].append(val)

    results: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        scores = np.array(boot_scores[key])
        if len(scores) == 0:
            results[key] = {
                "mean": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "std": float("nan"),
                "n_valid_boots": 0,
            }
        else:
            results[key] = {
                "mean": float(np.mean(scores)),
                "ci_lower": float(np.percentile(scores, 100 * alpha / 2)),
                "ci_upper": float(np.percentile(scores, 100 * (1 - alpha / 2))),
                "std": float(np.std(scores)),
                "n_valid_boots": int(len(scores)),
            }

    return results


def bootstrap_metric_difference(
    y_true: List[str],
    y_pred_ref: List[str],
    y_prob_ref: Optional[np.ndarray],
    y_pred_cmp: List[str],
    y_prob_cmp: Optional[np.ndarray],
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    pos_label: Optional[str] = None,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI for the *difference* in metrics between two models.

    Computes metric_ref − metric_cmp for each bootstrap sample.

    Returns:
        Dict keyed by metric → {mean_diff, ci_lower, ci_upper, std}.
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "f1", "roc_auc"]

    classes = sorted(set(y_true))
    if pos_label is None:
        pos_label = classes[-1] if len(classes) == 2 else classes[0]

    y_true_bin = np.array([1 if y == pos_label else 0 for y in y_true])
    y_pred_ref_bin = np.array([1 if y == pos_label else 0 for y in y_pred_ref])
    y_pred_cmp_bin = np.array([1 if y == pos_label else 0 for y in y_pred_cmp])

    alpha = 1.0 - confidence_level
    rng = np.random.RandomState(seed)
    n = len(y_true_bin)

    diffs: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true_bin[idx]

        if len(np.unique(yt)) < 2:
            continue

        yp_ref = y_pred_ref_bin[idx]
        ypr_ref = y_prob_ref[idx] if y_prob_ref is not None else None
        yp_cmp = y_pred_cmp_bin[idx]
        ypr_cmp = y_prob_cmp[idx] if y_prob_cmp is not None else None

        for key in metric_keys:
            val_ref = _compute_metric(yt, yp_ref, ypr_ref, key)
            val_cmp = _compute_metric(yt, yp_cmp, ypr_cmp, key)
            diffs[key].append(val_ref - val_cmp)

    results: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        d = np.array(diffs[key])
        if len(d) == 0:
            results[key] = {
                "mean_diff": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "std": float("nan"),
                "ci_excludes_zero": False,
                "n_valid_boots": 0,
            }
        else:
            lo = float(np.percentile(d, 100 * alpha / 2))
            hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
            results[key] = {
                "mean_diff": float(np.mean(d)),
                "ci_lower": lo,
                "ci_upper": hi,
                "std": float(np.std(d)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "n_valid_boots": int(len(d)),
            }

    return results


# ============================================================
# 4.  Orchestrator
# ============================================================

def run_statistical_tests(
    cv_results: Dict[str, Any],
    test_predictions: Dict[str, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all Phase 4 statistical significance tests.

    Args:
        cv_results:       Output of ``run_full_evaluation`` — must include
                          ``models.<name>.fold_metrics`` for Wilcoxon.
        test_predictions: Per-model predictions on the held-out test set.
                          Each entry: ``{y_true, y_pred, y_prob}``.
        config:           Config dict (reads ``statistical_tests`` section).

    Returns:
        Nested dict ready to be JSON-serialised to
        ``results/statistical_tests.json``.
    """
    if config is None:
        config = {}

    st_cfg = config.get("statistical_tests", {})
    n_boot = st_cfg.get("bootstrap_iterations", 1000)
    conf_level = st_cfg.get("confidence_level", 0.95)
    ref_model = st_cfg.get("reference_model", "neural")
    cmp_models = st_cfg.get("comparison_models", ["tfidf_lr", "embedding_lr"])
    seed = config.get("seed", 42)
    metric_keys = ["accuracy", "f1", "roc_auc"]

    output: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "bootstrap_iterations": n_boot,
            "confidence_level": conf_level,
            "reference_model": ref_model,
            "comparison_models": cmp_models,
            "seed": seed,
        },
        "mcnemar": {},
        "wilcoxon": {},
        "bootstrap_ci": {},
        "bootstrap_diff_ci": {},
    }

    ref_preds = test_predictions.get(ref_model)
    ref_cv = (cv_results.get("models", {}).get(ref_model) or {})

    # ---- McNemar tests --------------------------------------------------
    for cmp in cmp_models:
        cmp_preds = test_predictions.get(cmp)
        if ref_preds is None or cmp_preds is None:
            output["mcnemar"][f"{ref_model}_vs_{cmp}"] = {
                "skipped": True,
                "reason": "Missing predictions for one or both models.",
            }
            continue

        try:
            output["mcnemar"][f"{ref_model}_vs_{cmp}"] = mcnemar_test(
                y_true=ref_preds["y_true"],
                y_pred_a=ref_preds["y_pred"],
                y_pred_b=cmp_preds["y_pred"],
            )
        except Exception as e:
            output["mcnemar"][f"{ref_model}_vs_{cmp}"] = {
                "error": str(e),
            }

    # ---- Wilcoxon tests -------------------------------------------------
    for cmp in cmp_models:
        cmp_cv = (cv_results.get("models", {}).get(cmp) or {})
        ref_folds = ref_cv.get("fold_metrics")
        cmp_folds = cmp_cv.get("fold_metrics")

        if ref_folds is None or cmp_folds is None:
            output["wilcoxon"][f"{ref_model}_vs_{cmp}"] = {
                "skipped": True,
                "reason": "Missing fold-level metrics for one or both models.",
            }
            continue

        try:
            output["wilcoxon"][f"{ref_model}_vs_{cmp}"] = wilcoxon_fold_test(
                fold_metrics_a=ref_folds,
                fold_metrics_b=cmp_folds,
                metric_keys=metric_keys,
            )
        except Exception as e:
            output["wilcoxon"][f"{ref_model}_vs_{cmp}"] = {
                "error": str(e),
            }

    # ---- Bootstrap CIs per model ----------------------------------------
    for model_name, preds in test_predictions.items():
        try:
            output["bootstrap_ci"][model_name] = bootstrap_confidence_intervals(
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                y_prob=preds.get("y_prob"),
                n_boot=n_boot,
                confidence_level=conf_level,
                seed=seed,
                metric_keys=metric_keys,
            )
        except Exception as e:
            output["bootstrap_ci"][model_name] = {"error": str(e)}

    # ---- Bootstrap CIs for metric differences ---------------------------
    if ref_preds is not None:
        for cmp in cmp_models:
            cmp_preds = test_predictions.get(cmp)
            if cmp_preds is None:
                output["bootstrap_diff_ci"][f"{ref_model}_vs_{cmp}"] = {
                    "skipped": True,
                    "reason": "Missing predictions for comparison model.",
                }
                continue

            try:
                output["bootstrap_diff_ci"][f"{ref_model}_vs_{cmp}"] = (
                    bootstrap_metric_difference(
                        y_true=ref_preds["y_true"],
                        y_pred_ref=ref_preds["y_pred"],
                        y_prob_ref=ref_preds.get("y_prob"),
                        y_pred_cmp=cmp_preds["y_pred"],
                        y_prob_cmp=cmp_preds.get("y_prob"),
                        n_boot=n_boot,
                        confidence_level=conf_level,
                        seed=seed,
                        metric_keys=metric_keys,
                    )
                )
            except Exception as e:
                output["bootstrap_diff_ci"][f"{ref_model}_vs_{cmp}"] = {
                    "error": str(e),
                }

    return output


# ============================================================
# 5.  CLI entry point (standalone usage)
# ============================================================

def main():
    """Run statistical tests from saved evaluation artefacts."""
    ap = argparse.ArgumentParser(
        description="Phase 4 — Statistical significance tests"
    )
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument(
        "--results-dir", default=None,
        help="Override results directory (default: from config)",
    )
    args = ap.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = Path(
        args.results_dir
        or config.get("paths", {}).get("results_dir", "results")
    )

    cv_path = results_dir / "cv_results.json"
    test_path = results_dir / "test_results.json"
    predictions_path = results_dir / "test_predictions.json"

    if not cv_path.exists() or not predictions_path.exists():
        print(
            "Required files not found. Run eval_protocol.py first to "
            "generate cv_results.json and test_predictions.json."
        )
        return

    with open(cv_path, "r", encoding="utf-8") as f:
        cv_results = json.load(f)

    with open(predictions_path, "r", encoding="utf-8") as f:
        test_predictions_raw = json.load(f)

    # Convert y_prob lists back to numpy arrays
    test_predictions = {}
    for name, preds in test_predictions_raw.items():
        test_predictions[name] = {
            "y_true": preds["y_true"],
            "y_pred": preds["y_pred"],
            "y_prob": (
                np.array(preds["y_prob"]) if preds.get("y_prob") is not None
                else None
            ),
        }

    seed = config.get("seed", 42)
    set_deterministic(seed)

    stats = run_statistical_tests(cv_results, test_predictions, config)

    out_path = results_dir / "statistical_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Statistical tests saved → {out_path}")


if __name__ == "__main__":
    main()
