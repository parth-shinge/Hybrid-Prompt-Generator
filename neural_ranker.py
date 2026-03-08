"""
neural_ranker.py

Embedding-based Neural Ranker — exact architecture from the paper:
"A Hybrid Framework for Adaptive Prompt Generation Using Templates,
 LLMs, and Learned Rankers"

Architecture:
    Input(384) → Linear(384,128) → ReLU → Dropout(0.2)
               → Linear(128,64)  → ReLU
               → Linear(64,1)    → Sigmoid

Training:
    Loss:      Binary Cross Entropy
    Optimizer: Adam (lr=1e-3)
    Batch:     32
    Features:  Early stopping, checkpoint saving, train/val split

Public API:
    - PromptRankerNet       — the nn.Module
    - NeuralRankerTrainer   — trains and evaluates the model
    - train_ranker(texts, labels, config) → (metrics_dict, exp_dir)
    - predict(text, model_path, config)   → label
    - predict_proba(text, model_path, config) → (label, probability)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from seeds import set_deterministic

# Lazy-loaded sentence-transformers embedder
_EMBEDDER_CACHE: Dict[str, Any] = {}


# ============================================================
# 1.  Model architecture
# ============================================================

class PromptRankerNet(nn.Module):
    """Feed-forward binary ranker.

    Architecture (per paper):
        Linear(embed_dim, hidden_1) → ReLU → Dropout
        Linear(hidden_1, hidden_2)  → ReLU
        Linear(hidden_2, 1)         → Sigmoid
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_1: int = 128,
        hidden_2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, embed_dim) → (B, 1)
        return self.net(x)


# ============================================================
# 2.  Embedding helper
# ============================================================

def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return a sentence-transformers model (cached after first load)."""
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name, device="cpu")
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def _embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of texts into a (N, 384) float32 numpy array."""
    embedder = _get_embedder(model_name)
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ============================================================
# 3.  Config loader
# ============================================================

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config, returning a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _neural_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract neural_ranker subsection with safe defaults."""
    defaults = {
        "embed_model": "all-MiniLM-L6-v2",
        "embed_dim": 384,
        "hidden_1": 128,
        "hidden_2": 64,
        "dropout": 0.2,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
    }
    nr = config.get("neural_ranker", {})
    for k, v in defaults.items():
        nr.setdefault(k, v)
    return nr


# ============================================================
# 4.  Trainer
# ============================================================

class NeuralRankerTrainer:
    """Handles the full train → validate → checkpoint cycle.

    Integrates with:
        - ``utils.experiment.ExperimentTracker`` for metadata persistence
        - ``config.yaml`` for hyperparameter loading
        - SHAP interpretability (model exposes standard forward interface)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.nr = _neural_cfg(config)
        self.seed: int = config.get("seed", 42)
        self.test_split: float = config.get("test_split", 0.2)

        # Build model
        self.model = PromptRankerNet(
            embed_dim=self.nr["embed_dim"],
            hidden_1=self.nr["hidden_1"],
            hidden_2=self.nr["hidden_2"],
            dropout=self.nr["dropout"],
        )

        self.best_model_state: Optional[Dict] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

    # ---- public API ----

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        label_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping.

        Args:
            texts:  List of prompt texts.
            labels: List of string labels (e.g. "offline" / "gemini").
            label_map: Optional explicit mapping of label→{0,1}.
                       Auto-detected from sorted unique labels if omitted.

        Returns:
            Metrics dict with keys: best_epoch, best_val_loss, best_val_acc,
            train_loss_history, val_loss_history, val_acc_history, label_map.
        """
        set_deterministic(self.seed)

        # --- encode labels --------------------------------------------------
        if label_map is None:
            unique = sorted(set(labels))
            if len(unique) != 2:
                raise ValueError(f"Binary ranker requires exactly 2 classes, got {unique}")
            label_map = {u: i for i, u in enumerate(unique)}
        self._label_map = label_map
        y = np.array([label_map[l] for l in labels], dtype=np.float32)

        # --- embed -----------------------------------------------------------
        X = _embed_texts(texts, self.nr["embed_model"])

        # --- train / val split -----------------------------------------------
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * (1 - self.test_split))
        train_idx, val_idx = indices[:split], indices[split:]

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.nr["batch_size"], shuffle=True)

        # --- optimiser / loss ------------------------------------------------
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.nr["lr"])

        # --- training loop with early stopping -------------------------------
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.nr["epochs"] + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)
            self.history["train_loss"].append(epoch_loss)

            # --- validation ---------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = criterion(val_out, y_val).item()
                val_preds = (val_out >= 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # --- early stopping -----------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.nr["patience"]:
                    break

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": self.history["val_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
            "final_epoch": len(self.history["train_loss"]),
            "train_loss_history": self.history["train_loss"],
            "val_loss_history": self.history["val_loss"],
            "val_acc_history": self.history["val_acc"],
            "label_map": label_map,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save model + metadata to a ``.pt`` checkpoint.

        The checkpoint contains:
            - ``state_dict``: model weights
            - ``config``: neural ranker config subsection
            - ``label_map``: string-label → int mapping
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.nr,
                "label_map": getattr(self, "_label_map", None),
            },
            path,
        )
        return path


# ============================================================
# 5.  Public convenience functions
# ============================================================

def train_ranker(
    texts: List[str],
    labels: List[str],
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config.yaml",
) -> Tuple[Dict[str, Any], Path]:
    """High-level entry point: train the neural ranker and persist artifacts.

    Args:
        texts:  Training texts.
        labels: Training labels (binary).
        config: Pre-loaded config dict. Loaded from *config_path* if ``None``.
        config_path: Path to ``config.yaml``.

    Returns:
        (metrics_dict, experiment_dir_path)
    """
    if config is None:
        config = load_config(config_path)

    nr_cfg = _neural_cfg(config)

    # -- Experiment tracker --------------------------------------------------
    from utils.experiment import ExperimentTracker
    from utils.hashing import compute_data_sha256

    tracker = ExperimentTracker(
        config=config,
        seed=config.get("seed", 42),
        experiments_dir=config.get("paths", {}).get("experiments_dir", "experiments"),
    )
    tracker.set_dataset_hash(compute_data_sha256(list(zip(texts, labels))))

    # -- Train ---------------------------------------------------------------
    trainer = NeuralRankerTrainer(config)
    metrics = trainer.fit(texts, labels)

    # -- Persist -------------------------------------------------------------
    exp_dir = tracker.save_metadata()
    tracker.save_metrics(metrics)
    trainer.save_checkpoint(exp_dir / "model.pt")

    # Also save to models/ for easy loading
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(models_dir / "neural_ranker_best.pt")

    return metrics, exp_dir


def _load_checkpoint(model_path: Union[str, Path]) -> Tuple[PromptRankerNet, Dict[str, int]]:
    """Load a trained checkpoint and return (model, label_map)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PromptRankerNet(
        embed_dim=cfg.get("embed_dim", 384),
        hidden_1=cfg.get("hidden_1", 128),
        hidden_2=cfg.get("hidden_2", 64),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label_map: Dict[str, int] = ckpt.get("label_map") or {}
    return model, label_map


def predict(
    text: Union[str, List[str]],
    model_path: str = "models/neural_ranker_best.pt",
    config_path: str = "config.yaml",
) -> Union[str, List[str]]:
    """Return the predicted label(s) for one or more texts.

    Args:
        text: Single string or list of strings.
        model_path: Path to the ``.pt`` checkpoint.
        config_path: Path to ``config.yaml`` (used for embed model name).

    Returns:
        Predicted label string, or list of label strings.
    """
    config = load_config(config_path)
    nr = _neural_cfg(config)
    model, label_map = _load_checkpoint(model_path)
    inv_map = {v: k for k, v in label_map.items()}

    single = isinstance(text, str)
    texts = [text] if single else list(text)

    X = _embed_texts(texts, nr["embed_model"])
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()
    preds = (probs >= 0.5).astype(int)
    result = [inv_map.get(int(p), "unknown") for p in preds]
    return result[0] if single else result


def predict_proba(
    text: Union[str, List[str]],
    model_path: str = "models/neural_ranker_best.pt",
    config_path: str = "config.yaml",
) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
    """Return predicted label(s) with probabilities.

    Args:
        text: Single string or list of strings.
        model_path: Path to the ``.pt`` checkpoint.
        config_path: Path to ``config.yaml``.

    Returns:
        ``(label, probability)`` for a single text, or a list thereof.
    """
    config = load_config(config_path)
    nr = _neural_cfg(config)
    model, label_map = _load_checkpoint(model_path)
    inv_map = {v: k for k, v in label_map.items()}

    single = isinstance(text, str)
    texts = [text] if single else list(text)

    X = _embed_texts(texts, nr["embed_model"])
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()

    results = []
    for p in raw:
        idx = int(p >= 0.5)
        label = inv_map.get(idx, "unknown")
        prob = float(p) if idx == 1 else float(1 - p)
        results.append((label, prob))

    return results[0] if single else results
