"""
eval_protocol.py

Phase 3+4 — Formal Evaluation Protocol & Statistical Significance Testing
for the Hybrid Prompt Generator.

Implements:
    1. 5-fold stratified cross-validation for all models
    2. Held-out test split separate from CV
    3. Full metric suite: accuracy, precision, recall, F1, ROC-AUC
    4. JSON persistence to ``results/cv_results.json`` & ``results/test_results.json``
    5. Phase 4: Statistical significance tests (McNemar, Wilcoxon, bootstrap CIs)
       saved to ``results/statistical_tests.json``

Compatible with:
    - experiment logging (``utils.experiment.ExperimentTracker``)
    - neural ranker (``neural_ranker.py``)
    - config.yaml hyperparameters
    - baselines (``baselines.py``)
    - statistical_tests.py (Phase 4)

Usage:
    python eval_protocol.py
    python eval_protocol.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import yaml

from seeds import set_deterministic
from baselines import RandomBaseline, PopularityBaseline, TFIDFBaseline


# ============================================================
# 1.  Metrics computation
# ============================================================

def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_prob: Optional[np.ndarray] = None,
    pos_label: Optional[str] = None,
) -> Dict[str, float]:
    """Compute the full evaluation metric suite.

    Args:
        y_true:    Ground-truth label strings.
        y_pred:    Predicted label strings.
        y_prob:    (N,) array of predicted probabilities for the positive class.
                   If provided, ROC-AUC is computed; otherwise set to NaN.
        pos_label: The label considered "positive" for binary metrics.
                   Auto-detected as the second sorted class if omitted.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    classes = sorted(set(y_true))
    if pos_label is None:
        pos_label = classes[-1] if len(classes) == 2 else classes[0]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0.0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0.0)

    if y_prob is not None and len(classes) == 2:
        try:
            auc = roc_auc_score(
                [1 if y == pos_label else 0 for y in y_true],
                y_prob,
            )
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


# ============================================================
# 2.  Model wrappers (unified interface for CV)
# ============================================================

class _NeuralRankerWrapper:
    """Thin adapter so the neural ranker can be used in the CV loop
    with the same ``.fit / .predict / .predict_proba`` API as baselines.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._trainer = None
        self._label_map: Dict[str, int] = {}
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]) -> "_NeuralRankerWrapper":
        from neural_ranker import NeuralRankerTrainer

        self._trainer = NeuralRankerTrainer(self.config)
        self._trainer.fit(texts, labels)
        self._label_map = getattr(self._trainer, "_label_map", {})
        self.classes_ = np.array(sorted(self._label_map.keys()))
        return self

    def predict(self, texts: List[str]) -> List[str]:
        import torch
        from neural_ranker import _embed_texts, _neural_cfg

        nr = _neural_cfg(self.config)
        X = _embed_texts(texts, nr["embed_model"])
        self._trainer.model.eval()
        with torch.no_grad():
            probs = self._trainer.model(
                torch.tensor(X, dtype=torch.float32)
            ).squeeze(-1).numpy()
        inv = {v: k for k, v in self._label_map.items()}
        return [inv.get(int(p >= 0.5), "unknown") for p in probs]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        import torch
        from neural_ranker import _embed_texts, _neural_cfg

        nr = _neural_cfg(self.config)
        X = _embed_texts(texts, nr["embed_model"])
        self._trainer.model.eval()
        with torch.no_grad():
            p1 = self._trainer.model(
                torch.tensor(X, dtype=torch.float32)
            ).squeeze(-1).numpy()
        # Return (N, 2): [prob_class_0, prob_class_1]
        return np.column_stack([1 - p1, p1])


class _EmbeddingLogisticWrapper:
    """Wrapper around the existing embedding + LogReg ranker."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._clf = None
        self._scaler = None
        self.classes_: Optional[np.ndarray] = None
        self._embed_model = config.get("embedding_ranker", {}).get(
            "embed_model", "all-MiniLM-L6-v2"
        )

    def fit(self, texts: List[str], labels: List[str]) -> "_EmbeddingLogisticWrapper":
        from neural_ranker import _embed_texts
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        seed = self.config.get("seed", 42)
        set_deterministic(seed)
        X = _embed_texts(texts, self._embed_model)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        emb_cfg = self.config.get("embedding_ranker", {})
        self._clf = LogisticRegression(
            max_iter=emb_cfg.get("max_iter", 2000),
            solver=emb_cfg.get("solver", "liblinear"),
            random_state=seed,
        )
        self._clf.fit(Xs, labels)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, texts: List[str]) -> List[str]:
        from neural_ranker import _embed_texts

        X = _embed_texts(texts, self._embed_model)
        Xs = self._scaler.transform(X)
        return list(self._clf.predict(Xs))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        from neural_ranker import _embed_texts

        X = _embed_texts(texts, self._embed_model)
        Xs = self._scaler.transform(X)
        return self._clf.predict_proba(Xs)


# ============================================================
# 3.  Cross-validation engine
# ============================================================

def run_stratified_cv(
    model_factory,
    texts: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run stratified k-fold CV and return per-fold + aggregated metrics.

    Args:
        model_factory: Callable() → model with ``.fit / .predict / .predict_proba``.
        texts:  1-D array of text strings.
        labels: 1-D array of label strings.
        n_folds: Number of CV folds (default 5).
        seed: Random seed for fold splitting.

    Returns:
        Dict with ``fold_metrics`` (list of per-fold dicts) and
        ``mean_metrics`` (averaged across folds).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        set_deterministic(seed + fold_idx)

        X_train = [texts[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val = [texts[i] for i in val_idx]
        y_val = [labels[i] for i in val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        # Probability for positive class (second sorted class)
        classes = sorted(set(labels))
        pos_label = classes[-1] if len(classes) == 2 else classes[0]
        try:
            proba = model.predict_proba(X_val)
            pos_idx = list(model.classes_).index(pos_label)
            y_prob = proba[:, pos_idx]
        except Exception:
            y_prob = None

        metrics = compute_metrics(y_val, preds, y_prob=y_prob, pos_label=pos_label)
        metrics["fold"] = fold_idx + 1
        fold_metrics.append(metrics)

    # Aggregate
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mean_metrics = {}
    std_metrics = {}
    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics if not (isinstance(fm[k], float) and np.isnan(fm[k]))]
        mean_metrics[k] = float(np.mean(vals)) if vals else float("nan")
        std_metrics[k] = float(np.std(vals)) if vals else float("nan")

    return {
        "n_folds": n_folds,
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


# ============================================================
# 4.  Held-out test evaluation
# ============================================================

def run_held_out_test(
    model_factory,
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    seed: int = 42,
) -> Dict[str, float]:
    """Train on full train set and evaluate on held-out test set.

    Returns:
        Metrics dict (accuracy, precision, recall, f1, roc_auc).
    """
    set_deterministic(seed)

    model = model_factory()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    classes = sorted(set(y_train))
    pos_label = classes[-1] if len(classes) == 2 else classes[0]
    try:
        proba = model.predict_proba(X_test)
        pos_idx = list(model.classes_).index(pos_label)
        y_prob = proba[:, pos_idx]
    except Exception:
        y_prob = None

    return compute_metrics(y_test, preds, y_prob=y_prob, pos_label=pos_label)


# ============================================================
# 5.  Full evaluation pipeline
# ============================================================

def run_full_evaluation(
    texts: List[str],
    labels: List[str],
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config.yaml",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the complete Phase 3 evaluation protocol.

    Steps:
        1. Split data into train+CV / held-out test
        2. Run 5-fold stratified CV on train split for every model
        3. Train each model on full train split, evaluate on test
        4. Save results to ``results/cv_results.json`` and ``results/test_results.json``

    Args:
        texts:  All text samples.
        labels: All label samples.
        config: Pre-loaded config dict. Loaded from *config_path* if None.
        config_path: Path to config.yaml.

    Returns:
        (cv_results, test_results) dicts.
    """
    if config is None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    n_folds = config.get("cv_folds", 5)
    test_split = config.get("test_split", 0.2)
    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(seed)

    # ---- Stratified train / test split ----------------------------------
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    n = len(texts_arr)
    indices = np.arange(n)

    # Use StratifiedKFold with 1/test_split folds to get a proper stratified split
    n_splits_for_test = max(2, int(round(1.0 / test_split)))
    splitter = StratifiedKFold(n_splits=n_splits_for_test, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(texts_arr, labels_arr))

    X_train = texts_arr[train_idx]
    y_train = labels_arr[train_idx]
    X_test = texts_arr[test_idx]
    y_test = labels_arr[test_idx]

    print(f"Dataset: {n} total → {len(X_train)} train, {len(X_test)} test")
    print(f"  Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test  class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ---- Define model factories -----------------------------------------
    model_factories = {
        "random": lambda: RandomBaseline(seed=seed),
        "popularity": lambda: PopularityBaseline(),
        "tfidf_lr": lambda: TFIDFBaseline.from_config(config),
    }

    # Embedding LogReg — only if sentence-transformers available
    try:
        from neural_ranker import _get_embedder
        model_factories["embedding_lr"] = lambda: _EmbeddingLogisticWrapper(config)
    except Exception:
        print("  [skip] embedding_lr — sentence-transformers not available")

    # Neural ranker — only if torch + sentence-transformers available
    try:
        import torch
        from neural_ranker import PromptRankerNet
        model_factories["neural"] = lambda: _NeuralRankerWrapper(config)
    except Exception:
        print("  [skip] neural — torch/sentence-transformers not available")

    # ---- 5-fold stratified CV on train split ----------------------------
    print(f"\n{'='*60}")
    print(f"  5-Fold Stratified Cross-Validation (seed={seed})")
    print(f"{'='*60}")

    cv_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_folds": n_folds,
        "train_size": int(len(X_train)),
        "models": {},
    }

    for name, factory in model_factories.items():
        print(f"\n  [{name}]")
        try:
            result = run_stratified_cv(
                factory,
                X_train,
                y_train,
                n_folds=n_folds,
                seed=seed,
            )
            cv_results["models"][name] = result
            m = result["mean_metrics"]
            print(f"    Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
        except Exception as e:
            cv_results["models"][name] = {"error": str(e)}
            print(f"    ERROR: {e}")

    # ---- Held-out test evaluation (with prediction capture) --------------
    print(f"\n{'='*60}")
    print(f"  Held-Out Test Evaluation")
    print(f"{'='*60}")

    test_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "models": {},
    }

    # Collect raw predictions for Phase 4 statistical tests
    test_predictions: Dict[str, Dict[str, Any]] = {}

    for name, factory in model_factories.items():
        print(f"\n  [{name}]")
        try:
            set_deterministic(seed)
            model = factory()
            model.fit(list(X_train), list(y_train))
            preds = model.predict(list(X_test))

            classes = sorted(set(y_train))
            pos_label = classes[-1] if len(classes) == 2 else classes[0]
            try:
                proba = model.predict_proba(list(X_test))
                pos_idx = list(model.classes_).index(pos_label)
                y_prob = proba[:, pos_idx]
            except Exception:
                y_prob = None

            result = compute_metrics(
                list(y_test), preds, y_prob=y_prob, pos_label=pos_label,
            )
            test_results["models"][name] = result

            # Store raw predictions for statistical tests
            test_predictions[name] = {
                "y_true": list(y_test),
                "y_pred": list(preds),
                "y_prob": y_prob.tolist() if y_prob is not None else None,
            }

            print(f"    Acc={result['accuracy']:.4f}  P={result['precision']:.4f}  "
                  f"R={result['recall']:.4f}  F1={result['f1']:.4f}  AUC={result['roc_auc']:.4f}")
        except Exception as e:
            test_results["models"][name] = {"error": str(e)}
            print(f"    ERROR: {e}")

    # ---- Save results ---------------------------------------------------
    cv_path = results_dir / "cv_results.json"
    test_path = results_dir / "test_results.json"
    predictions_path = results_dir / "test_predictions.json"

    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\n  Saved CV results       → {cv_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Saved test results     → {test_path}")

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2, default=str)
    print(f"  Saved test predictions → {predictions_path}")

    # ---- Phase 4: Statistical significance tests ------------------------
    stat_results = None
    try:
        from statistical_tests import run_statistical_tests

        # Convert y_prob lists back to numpy arrays for the stats module
        preds_for_stats: Dict[str, Dict[str, Any]] = {}
        for mname, mpreds in test_predictions.items():
            preds_for_stats[mname] = {
                "y_true": mpreds["y_true"],
                "y_pred": mpreds["y_pred"],
                "y_prob": (
                    np.array(mpreds["y_prob"])
                    if mpreds.get("y_prob") is not None else None
                ),
            }

        stat_results = run_statistical_tests(
            cv_results, preds_for_stats, config,
        )

        stat_path = results_dir / "statistical_tests.json"
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat_results, f, indent=2, default=str)
        print(f"  Saved stat tests       → {stat_path}")
    except Exception as e:
        print(f"  [warn] Statistical tests skipped: {e}")

    # ---- Experiment tracking --------------------------------------------
    try:
        from utils.experiment import ExperimentTracker
        from utils.hashing import compute_data_sha256

        tracker = ExperimentTracker(
            config=config,
            seed=seed,
            experiments_dir=config.get("paths", {}).get("experiments_dir", "experiments"),
        )
        tracker.set_dataset_hash(compute_data_sha256(list(zip(texts, labels))))
        exp_dir = tracker.save_metadata()
        metrics_payload: Dict[str, Any] = {
            "cv_results": cv_results,
            "test_results": test_results,
        }
        if stat_results is not None:
            metrics_payload["statistical_tests"] = stat_results
        tracker.save_metrics(metrics_payload)
        print(f"  Experiment logged      → {exp_dir}")
    except Exception as e:
        print(f"  [warn] Experiment logging skipped: {e}")

    return cv_results, test_results


# ============================================================
# 6.  CLI entry point
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Phase 3 — Full evaluation protocol for the Hybrid Prompt Generator"
    )
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    # Load dataset
    from database import get_choice_dataset

    rows = get_choice_dataset()
    if not rows:
        print("No dataset available. Generate variant pairs and choose in the app first.")
        return

    texts, labels = zip(*rows)
    print(f"Loaded {len(texts)} samples from the choices dataset.")

    run_full_evaluation(list(texts), list(labels), config_path=args.config)


if __name__ == "__main__":
    main()
