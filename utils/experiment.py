"""
utils/experiment.py

Experiment tracking and metadata persistence.

Creates a timestamped experiment directory under ``experiments/``
and writes a comprehensive ``metadata.json`` alongside config
snapshots, model checkpoints, and metrics.

Directory layout produced per run:
    experiments/<timestamp>/
        metadata.json
        config_snapshot.yaml
        model.pt          (saved externally by caller)
        metrics.json

Usage:
    from utils.experiment import ExperimentTracker
    tracker = ExperimentTracker(config_dict, seed=42)
    tracker.set_dataset_hash("abcdef...")
    exp_dir = tracker.save_metadata()        # returns Path
    tracker.save_metrics({"loss": 0.12, "acc": 0.93})
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.git_info import get_git_commit, is_dirty


class ExperimentTracker:
    """Captures and persists all metadata for a single experiment run."""

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int = 42,
        experiments_dir: str = "experiments",
    ) -> None:
        self.experiment_id: str = uuid.uuid4().hex[:12]
        self.timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.seed: int = seed
        self.config: Dict[str, Any] = config
        self.experiments_dir: str = experiments_dir

        self.dataset_hash: Optional[str] = None
        self.git_commit: str = get_git_commit()
        self.git_dirty: Optional[bool] = is_dirty()

        # Build experiment path
        self._exp_dir: Path = Path(experiments_dir) / self.timestamp

    @property
    def exp_dir(self) -> Path:
        """Return the experiment directory path (creates it lazily)."""
        return self._exp_dir

    # ---- Setters ----

    def set_dataset_hash(self, hash_hex: str) -> None:
        """Store the SHA-256 hash of the dataset used for training."""
        self.dataset_hash = hash_hex

    # ---- Persistence ----

    def save_metadata(self, base_dir: Optional[str] = None) -> Path:
        """Write ``metadata.json`` and ``config_snapshot.yaml`` to the experiment dir.

        Args:
            base_dir: Override for the experiments base directory.

        Returns:
            Path to the experiment directory.
        """
        if base_dir:
            self._exp_dir = Path(base_dir) / self.timestamp

        self._exp_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "dataset_hash": self.dataset_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config": self.config,
        }

        meta_path = self._exp_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Also save a standalone config snapshot
        cfg_path = self._exp_dir / "config_snapshot.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        return self._exp_dir

    def save_metrics(self, metrics: Dict[str, Any], base_dir: Optional[str] = None) -> Path:
        """Write ``metrics.json`` to the experiment directory.

        Args:
            metrics: Dictionary of metric names to values.
            base_dir: Override for the experiments base directory.

        Returns:
            Path to the saved metrics file.
        """
        if base_dir:
            self._exp_dir = Path(base_dir) / self.timestamp

        self._exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self._exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        return metrics_path

    def save_model(self, model_obj: Any, base_dir: Optional[str] = None) -> Path:
        """Save a PyTorch model's ``state_dict`` to the experiment directory.

        Args:
            model_obj: A ``torch.nn.Module`` instance.
            base_dir: Override for the experiments base directory.

        Returns:
            Path to the saved ``model.pt`` file.
        """
        import torch

        if base_dir:
            self._exp_dir = Path(base_dir) / self.timestamp

        self._exp_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._exp_dir / "model.pt"
        torch.save(model_obj.state_dict(), model_path)
        return model_path
