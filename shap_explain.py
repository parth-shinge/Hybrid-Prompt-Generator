"""
shap_explain.py

Phase 6 — SHAP Interpretability for the Hybrid Prompt Generator.

Provides interpretable explanations for the neural ranker predictions
using SHAP (SHapley Additive exPlanations).

The neural ranker (``PromptRankerNet``) operates on 384-dim
sentence-transformer embeddings.  SHAP values are computed over those
embedding dimensions using ``DeepExplainer`` (with ``KernelExplainer``
as an automatic fallback).

Public API:
    - compute_global_shap   → global feature importance across a dataset
    - compute_local_shap    → per-prediction feature contributions
    - plot_global_importance → matplotlib bar chart figure
    - plot_local_explanation → matplotlib waterfall-style figure

Compatible with:
    - neural_ranker.py   (model loading / embedding)
    - eval_protocol.py   (evaluation pipeline)
    - config.yaml        (shap section)
    - artifacts/         (optional experiment logging)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1.  Internal helpers
# ============================================================

def _load_model_and_embed(
    texts: List[str],
    model_path: str = "models/neural_ranker_best.pt",
    config_path: str = "config.yaml",
):
    """Load the neural ranker checkpoint, embed texts, return (model, embeddings, label_map, config)."""
    from neural_ranker import _load_checkpoint, _embed_texts, load_config, _neural_cfg

    config = load_config(config_path)
    nr = _neural_cfg(config)
    model, label_map = _load_checkpoint(model_path)
    model.eval()

    embeddings = _embed_texts(texts, nr["embed_model"])
    return model, embeddings, label_map, config


def _shap_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ``shap`` subsection from config with safe defaults."""
    defaults = {
        "background_samples": 50,
        "max_display_dims": 20,
        "model_path": "models/neural_ranker_best.pt",
    }
    shap_cfg = config.get("shap", {})
    for k, v in defaults.items():
        shap_cfg.setdefault(k, v)
    return shap_cfg


def _ensure_results_dir(results_dir: str = "results") -> Path:
    """Create results directory if it doesn't exist."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# 2.  Global SHAP Feature Importance
# ============================================================

def compute_global_shap(
    texts: List[str],
    model_path: str = "models/neural_ranker_best.pt",
    config_path: str = "config.yaml",
    max_samples: Optional[int] = None,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Compute global SHAP feature importance across a dataset.

    Uses ``DeepExplainer`` for the PyTorch neural ranker.  Falls back
    to ``KernelExplainer`` if ``DeepExplainer`` raises an error.

    Args:
        texts:        List of prompt texts to explain.
        model_path:   Path to the ``.pt`` checkpoint.
        config_path:  Path to ``config.yaml``.
        max_samples:  Cap on the number of texts to use (for speed).
        results_dir:  Directory for saving ``shap_global.json``.

    Returns:
        Dict with keys: ``mean_abs_shap`` (list of floats, one per dim),
        ``top_dims`` (list of ``{dim, importance}`` dicts, descending),
        ``n_samples``, ``timestamp``.
    """
    import shap
    import torch

    # Subsample if requested
    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in indices]

    model, X, label_map, config = _load_model_and_embed(texts, model_path, config_path)
    shap_cfg = _shap_config(config)
    max_display = shap_cfg.get("max_display_dims", 20)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Background data for the explainer
    bg_size = min(shap_cfg.get("background_samples", 50), len(X))
    background = X_tensor[:bg_size]

    # Try DeepExplainer first, fallback to KernelExplainer
    shap_values = None
    explainer_used = "DeepExplainer"

    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_tensor)
    except Exception as e:
        logger.warning("DeepExplainer failed (%s), falling back to KernelExplainer", e)
        explainer_used = "KernelExplainer"
        try:
            def _model_fn(x):
                with torch.no_grad():
                    return model(torch.tensor(x, dtype=torch.float32)).numpy()

            explainer = shap.KernelExplainer(_model_fn, background.numpy())
            shap_values = explainer.shap_values(X, nsamples=100)
        except Exception as e2:
            logger.error("KernelExplainer also failed: %s", e2)
            raise RuntimeError(f"Both SHAP explainers failed: DeepExplainer({e}), KernelExplainer({e2})")

    # shap_values may be a list (one per output) — take first if so
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_arr = np.array(shap_values)  # (N, embed_dim)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)
    mean_abs = np.mean(np.abs(shap_arr), axis=0).flatten()  # (embed_dim,)

    # Build ranked list of dimensions
    ranked_indices = np.argsort(mean_abs)[::-1].flatten().tolist()
    mean_abs_list = mean_abs.flatten().tolist()
    top_dims = [
        {"dim": idx, "label": f"dim_{idx}", "importance": round(float(mean_abs_list[idx]), 6)}
        for idx in ranked_indices[:max_display]
    ]

    result = {
        "mean_abs_shap": [round(float(v), 6) for v in mean_abs_list],
        "top_dims": top_dims,
        "n_samples": len(texts),
        "embed_dim": int(shap_arr.shape[1]),
        "explainer": explainer_used,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save to results/
    try:
        out_path = _ensure_results_dir(results_dir) / "shap_global.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("Global SHAP results saved to %s", out_path)
    except Exception as e:
        logger.warning("Could not save global SHAP results: %s", e)

    return result


# ============================================================
# 3.  Local SHAP Explanations
# ============================================================

def compute_local_shap(
    text: Union[str, List[str]],
    model_path: str = "models/neural_ranker_best.pt",
    config_path: str = "config.yaml",
    background_texts: Optional[List[str]] = None,
    n_background: int = 50,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Generate SHAP explanations for individual prediction(s).

    Args:
        text:             Single prompt string or list of prompts.
        model_path:       Path to the ``.pt`` checkpoint.
        config_path:      Path to ``config.yaml``.
        background_texts: Texts to use as SHAP background.  If ``None``,
                          a small synthetic background is generated.
        n_background:     Number of background samples to use.
        results_dir:      Directory for saving ``shap_local.json``.

    Returns:
        Dict with keys: ``predictions`` (list of dicts with label/prob/
        shap_values/top_contributions per text).
    """
    import shap
    import torch
    from neural_ranker import _load_checkpoint, _embed_texts, load_config, _neural_cfg

    single = isinstance(text, str)
    texts = [text] if single else list(text)

    config = load_config(config_path)
    nr = _neural_cfg(config)
    shap_cfg = _shap_config(config)
    max_display = shap_cfg.get("max_display_dims", 20)

    model, label_map = _load_checkpoint(model_path)
    model.eval()
    inv_map = {v: k for k, v in label_map.items()}

    # Embed the target texts
    X = _embed_texts(texts, nr["embed_model"])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Build background
    if background_texts is not None:
        bg_X = _embed_texts(background_texts[:n_background], nr["embed_model"])
    else:
        # Use small random noise as background when no texts provided
        rng = np.random.RandomState(42)
        bg_X = rng.randn(min(n_background, 20), X.shape[1]).astype(np.float32)
    bg_tensor = torch.tensor(bg_X, dtype=torch.float32)

    # Compute SHAP values
    shap_values = None
    explainer_used = "DeepExplainer"
    try:
        explainer = shap.DeepExplainer(model, bg_tensor)
        shap_values = explainer.shap_values(X_tensor)
    except Exception as e:
        logger.warning("DeepExplainer failed (%s), falling back to KernelExplainer", e)
        explainer_used = "KernelExplainer"
        try:
            def _model_fn(x):
                with torch.no_grad():
                    return model(torch.tensor(x, dtype=torch.float32)).numpy()

            explainer = shap.KernelExplainer(_model_fn, bg_X)
            shap_values = explainer.shap_values(X, nsamples=100)
        except Exception as e2:
            raise RuntimeError(f"Both SHAP explainers failed: DeepExplainer({e}), KernelExplainer({e2})")

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_arr = np.array(shap_values)  # (N, embed_dim)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    # Model predictions
    with torch.no_grad():
        probs = model(X_tensor).squeeze(-1).numpy()
    probs = np.atleast_1d(probs)

    predictions = []
    for i, txt in enumerate(texts):
        sv = shap_arr[i].flatten()  # (embed_dim,)
        prob = float(probs[i].item()) if hasattr(probs[i], 'item') else float(probs[i])
        pred_idx = int(prob >= 0.5)
        label = inv_map.get(pred_idx, "unknown")

        # Top contributing dimensions (by absolute SHAP value)
        sv_list = sv.flatten().tolist()
        ranked = np.argsort(np.abs(sv))[::-1].flatten().tolist()
        top_contributions = [
            {
                "dim": idx,
                "label": f"dim_{idx}",
                "shap_value": round(float(sv_list[idx]), 6),
                "abs_value": round(abs(float(sv_list[idx])), 6),
            }
            for idx in ranked[:max_display]
        ]

        predictions.append({
            "text": txt[:200],  # truncate for readability
            "predicted_label": label,
            "probability": round(prob, 4),
            "shap_values": [round(float(v), 6) for v in sv_list],
            "top_contributions": top_contributions,
        })

    result = {
        "predictions": predictions,
        "explainer": explainer_used,
        "n_background": len(bg_X),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save to results/
    try:
        out_path = _ensure_results_dir(results_dir) / "shap_local.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("Local SHAP results saved to %s", out_path)
    except Exception as e:
        logger.warning("Could not save local SHAP results: %s", e)

    return result


# ============================================================
# 4.  Visualization — Global Importance
# ============================================================

def plot_global_importance(
    global_result: Dict[str, Any],
    max_dims: int = 20,
):
    """Create a matplotlib bar chart of the top embedding dimensions.

    Args:
        global_result: Dict returned by ``compute_global_shap``.
        max_dims:      Max number of dimensions to show.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    top_dims = global_result.get("top_dims", [])[:max_dims]
    if not top_dims:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    labels = [d["label"] for d in top_dims]
    values = [d["importance"] for d in top_dims]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))

    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # highest importance at top
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(
        f"Global Feature Importance — Top {len(labels)} Embedding Dimensions",
        fontsize=12, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


# ============================================================
# 5.  Visualization — Local Explanation
# ============================================================

def plot_local_explanation(
    local_prediction: Dict[str, Any],
    max_dims: int = 15,
):
    """Create a waterfall-style bar chart for a single prediction.

    Args:
        local_prediction: Single entry from ``compute_local_shap``'s
                          ``predictions`` list.
        max_dims:         Max number of dimensions to display.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    top = local_prediction.get("top_contributions", [])[:max_dims]
    if not top:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    labels = [d["label"] for d in top]
    values = [d["shap_value"] for d in top]
    label = local_prediction.get("predicted_label", "?")
    prob = local_prediction.get("probability", 0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))
    y_pos = np.arange(len(labels))

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=11)
    ax.set_title(
        f"Local Explanation — Predicted: {label} (p={prob:.3f})",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(x=0, color="grey", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig
