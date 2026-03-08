"""
ensemble_synthesis.py

Phase 5 — Ensemble Prompt Synthesis for the Hybrid Prompt Generator.

Combines template-based (offline) and Gemini-generated prompts using a
scoring-based ensemble mechanism:

    1. Slot coverage score  — do template slots appear in the text?
    2. Fluency score        — heuristic length / repetition / readability
    3. Weighted ensemble    — final = α·slot + β·fluency

The highest-scoring variant is selected as the final output.

Compatible with:
    - prompt_generator.py  (hybrid generation flow)
    - config.yaml           (ensemble section)
    - artifacts/             (optional score logging)

Usage:
    from ensemble_synthesis import ensemble_select
    source, text, log = ensemble_select(offline_text, gemini_text, slots, config)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1.  Slot Coverage Scoring
# ============================================================

def compute_slot_coverage(
    prompt: str,
    slots: Dict[str, Optional[str]],
) -> float:
    """Score how many required slot values appear in the prompt.

    Performs case-insensitive substring matching for each non-empty
    slot value.  Returns a normalised score in [0, 1].

    Args:
        prompt: The generated prompt text.
        slots:  Dict mapping slot names to their values.  ``None`` or
                empty-string values are ignored (not counted).

    Returns:
        Float in [0, 1].  1.0 means every non-empty slot value was
        found in the prompt text.
    """
    active_slots = {
        k: v for k, v in slots.items()
        if v is not None and str(v).strip()
    }
    if not active_slots:
        return 1.0  # nothing to cover → perfect score

    prompt_lower = prompt.lower()
    hits = sum(
        1 for v in active_slots.values()
        if str(v).strip().lower() in prompt_lower
    )
    return hits / len(active_slots)


# ============================================================
# 2.  Fluency Scoring
# ============================================================

def _length_score(text: str, min_len: int = 50, max_len: int = 500) -> float:
    """Reward prompts in a sweet-spot character range.

    Returns 1.0 if length is within [min_len, max_len], tapers
    linearly outside that window down to 0.0.
    """
    n = len(text)
    if n == 0:
        return 0.0
    if min_len <= n <= max_len:
        return 1.0
    if n < min_len:
        return max(0.0, n / min_len)
    # n > max_len — gentle taper: halve score at 2×max_len
    return max(0.0, 1.0 - (n - max_len) / max_len)


def _repetition_penalty(text: str) -> float:
    """Penalise repeated bigrams/trigrams.

    Returns a score in [0, 1] where 1.0 = no repetition,
    0.0 = extreme repetition.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < 4:
        return 1.0  # too short to judge

    def _ngram_ratio(tokens: List[str], n: int) -> float:
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(ngrams)

    bi_rep = _ngram_ratio(words, 2)
    tri_rep = _ngram_ratio(words, 3)
    penalty = 1.0 - min(1.0, (bi_rep + tri_rep) / 2)
    return max(0.0, penalty)


def _readability_proxy(text: str) -> float:
    """Simple readability proxy based on sentence structure.

    Rewards prompts with:
      - Multiple sentences (but not excessively many)
      - Moderate average sentence length (8–25 words)
      - Moderate proportion of long words (≥ 7 chars)

    Returns a score in [0, 1].
    """
    # Split into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = max(1, len(sentences))

    words = re.findall(r"\w+", text)
    n_words = max(1, len(words))

    avg_sent_len = n_words / n_sentences

    # Sentence count bonus: 2-5 sentences is ideal
    if 2 <= n_sentences <= 5:
        sent_score = 1.0
    elif n_sentences == 1:
        sent_score = 0.6
    else:
        sent_score = max(0.3, 1.0 - (n_sentences - 5) * 0.1)

    # Average sentence length: 8-25 words ideal
    if 8 <= avg_sent_len <= 25:
        len_score = 1.0
    elif avg_sent_len < 8:
        len_score = max(0.3, avg_sent_len / 8)
    else:
        len_score = max(0.3, 1.0 - (avg_sent_len - 25) / 25)

    # Long word proportion: 15-40% ideal
    long_words = sum(1 for w in words if len(w) >= 7)
    long_ratio = long_words / n_words
    if 0.15 <= long_ratio <= 0.40:
        vocab_score = 1.0
    elif long_ratio < 0.15:
        vocab_score = max(0.5, long_ratio / 0.15)
    else:
        vocab_score = max(0.3, 1.0 - (long_ratio - 0.40) / 0.40)

    return (sent_score + len_score + vocab_score) / 3


def compute_fluency_score(
    prompt: str,
    min_length: int = 50,
    max_length: int = 500,
) -> float:
    """Compute a composite fluency score for a prompt.

    Combines length appropriateness, repetition penalty, and a
    readability proxy.

    Args:
        prompt:      The prompt text.
        min_length:  Minimum ideal character length.
        max_length:  Maximum ideal character length.

    Returns:
        Float in [0, 1].
    """
    if not prompt or not prompt.strip():
        return 0.0

    ls = _length_score(prompt, min_length, max_length)
    rp = _repetition_penalty(prompt)
    rd = _readability_proxy(prompt)

    # Weighted combination of sub-scores
    return 0.35 * ls + 0.35 * rp + 0.30 * rd


# ============================================================
# 3.  Weighted Ensemble Score
# ============================================================

def compute_ensemble_score(
    slot_score: float,
    fluency_score: float,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> float:
    """Compute the final ensemble score.

    ``final = alpha * slot_score + beta * fluency_score``

    Args:
        slot_score:    Slot coverage score in [0, 1].
        fluency_score: Fluency score in [0, 1].
        alpha:         Weight for slot coverage.
        beta:          Weight for fluency.

    Returns:
        Float (weighted sum, typically in [0, 1]).
    """
    return alpha * slot_score + beta * fluency_score


# ============================================================
# 4.  Multi-Candidate Selection
# ============================================================

def select_best_prompt(
    candidates: List[Dict[str, str]],
    slots: Dict[str, Optional[str]],
    alpha: float = 0.6,
    beta: float = 0.4,
    min_length: int = 50,
    max_length: int = 500,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Score every candidate and select the best one.

    Args:
        candidates: List of ``{"source": ..., "text": ...}`` dicts.
        slots:      Slot name → value mapping for coverage scoring.
        alpha:      Slot-coverage weight.
        beta:       Fluency weight.
        min_length: Min ideal prompt length (chars).
        max_length: Max ideal prompt length (chars).

    Returns:
        (best_candidate, scoring_log) where *scoring_log* is a list
        of per-candidate score dicts.
    """
    scoring_log: List[Dict[str, Any]] = []
    best_idx = 0
    best_score = -1.0

    for i, cand in enumerate(candidates):
        text = cand.get("text", "")
        slot_sc = compute_slot_coverage(text, slots)
        fluency_sc = compute_fluency_score(text, min_length, max_length)
        final_sc = compute_ensemble_score(slot_sc, fluency_sc, alpha, beta)

        entry = {
            "source": cand.get("source", f"candidate_{i}"),
            "slot_score": round(slot_sc, 4),
            "fluency_score": round(fluency_sc, 4),
            "final_score": round(final_sc, 4),
        }
        scoring_log.append(entry)

        if final_sc > best_score:
            best_score = final_sc
            best_idx = i

    scoring_log[best_idx]["selected"] = True
    return candidates[best_idx], scoring_log


# ============================================================
# 5.  Convenience Function for Hybrid Flow
# ============================================================

def ensemble_select(
    offline_text: str,
    gemini_text: str,
    slots: Dict[str, Optional[str]],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Select the best prompt between offline and Gemini variants.

    This is the main entry point called from ``prompt_generator.py``.

    Args:
        offline_text: Template-generated prompt.
        gemini_text:  Gemini-generated prompt.
        slots:        Mapping of slot names to their user-provided values
                      (e.g. ``{"tool": "Canva", "topic": "...", ...}``).
        config:       Full config dict (reads ``ensemble`` section).

    Returns:
        ``(selected_source, selected_text, scoring_log)``
        where *selected_source* is ``"offline"`` or ``"gemini"``.
    """
    if config is None:
        config = {}

    ens_cfg = config.get("ensemble", {})
    alpha = ens_cfg.get("alpha", 0.6)
    beta = ens_cfg.get("beta", 0.4)
    min_length = ens_cfg.get("min_length", 50)
    max_length = ens_cfg.get("max_length", 500)
    log_scores = ens_cfg.get("log_scores", True)

    candidates = [
        {"source": "offline", "text": offline_text},
        {"source": "gemini", "text": gemini_text},
    ]

    best, scoring_log = select_best_prompt(
        candidates, slots,
        alpha=alpha, beta=beta,
        min_length=min_length, max_length=max_length,
    )

    # Optional: persist scoring log to artifacts
    if log_scores:
        try:
            os.makedirs("artifacts", exist_ok=True)
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "slots": {k: v for k, v in slots.items() if v},
                "scores": scoring_log,
                "selected_prompt": best["source"],
            }
            with open(
                os.path.join("artifacts", "ensemble_log.jsonl"),
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(json.dumps(log_entry, default=str) + "\n")
        except Exception:
            pass  # logging failure must never break generation

    return best["source"], best["text"], scoring_log
