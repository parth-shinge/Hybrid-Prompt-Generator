"""
baselines.py

Baseline models for the evaluation protocol (Phase 3).

Provides three baselines for comparison against the learned rankers:

1. RandomBaseline     — predicts uniformly at random
2. PopularityBaseline — always predicts the most frequent label
3. TFIDFBaseline      — TF-IDF + LogisticRegression (sklearn pipeline)

All baselines implement a common interface:
    .fit(texts, labels)
    .predict(texts)       → list of label strings
    .predict_proba(texts) → np.ndarray of shape (N, 2) with class probabilities
    .classes_             → array of class label strings
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from seeds import set_deterministic


# ============================================================
# 1.  Random baseline
# ============================================================

class RandomBaseline:
    """Predicts uniformly at random among the training labels.

    This establishes the floor performance: any useful model must
    beat ~50 % accuracy on a balanced binary dataset.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]) -> "RandomBaseline":
        self.classes_ = np.array(sorted(set(labels)))
        return self

    def predict(self, texts: List[str]) -> List[str]:
        rng = np.random.RandomState(self.seed)
        return [rng.choice(self.classes_) for _ in texts]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        k = len(self.classes_)
        # Uniform probability for each class
        return np.full((n, k), 1.0 / k, dtype=np.float64)


# ============================================================
# 2.  Popularity (majority-class) baseline
# ============================================================

class PopularityBaseline:
    """Always predicts the most frequent label from training data.

    On a balanced dataset this is identical to random; on imbalanced
    data it reveals how far accuracy can go with zero signal.
    """

    def __init__(self) -> None:
        self.majority_label_: Optional[str] = None
        self.classes_: Optional[np.ndarray] = None
        self._class_probs: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]) -> "PopularityBaseline":
        self.classes_ = np.array(sorted(set(labels)))
        counts = {c: 0 for c in self.classes_}
        for l in labels:
            counts[l] += 1
        total = len(labels)
        self.majority_label_ = max(counts, key=counts.get)
        self._class_probs = np.array([counts[c] / total for c in self.classes_])
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_label_] * len(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        return np.tile(self._class_probs, (n, 1))


# ============================================================
# 3.  TF-IDF + Logistic Regression baseline
# ============================================================

class TFIDFBaseline:
    """TF-IDF vectorization + Logistic Regression.

    This is the simplest feature-engineering approach and serves as
    the natural baseline before moving to embedding-based models.
    Hyperparameters are loaded from config.yaml's ``tfidf_ranker`` section.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        solver: str = "liblinear",
        max_iter: int = 2000,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.pipeline = make_pipeline(
            TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
            LogisticRegression(
                max_iter=max_iter,
                solver=solver,
                random_state=seed,
            ),
        )
        self.classes_: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config: dict) -> "TFIDFBaseline":
        """Construct from a config.yaml dict."""
        tfidf_cfg = config.get("tfidf_ranker", {})
        seed = config.get("seed", 42)
        ngram = tfidf_cfg.get("ngram_range", [1, 2])
        return cls(
            max_features=tfidf_cfg.get("max_features", 5000),
            ngram_range=tuple(ngram),
            solver=tfidf_cfg.get("solver", "liblinear"),
            max_iter=tfidf_cfg.get("max_iter", 2000),
            seed=seed,
        )

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFBaseline":
        set_deterministic(self.seed)
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline[-1].classes_
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)
