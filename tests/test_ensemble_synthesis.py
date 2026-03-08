"""
tests/test_ensemble_synthesis.py

Unit tests for Phase 5 — Ensemble Prompt Synthesis.

Tests:
    - Slot coverage scoring (full, partial, empty)
    - Fluency scoring (short, optimal, long, repetitive)
    - Ensemble score weighted combination
    - Multi-candidate selection
    - End-to-end ensemble_select convenience function
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensemble_synthesis import (
    compute_slot_coverage,
    compute_fluency_score,
    compute_ensemble_score,
    select_best_prompt,
    ensemble_select,
)


# ---- Slot Coverage ----

class TestSlotCoverage:
    def test_full_coverage(self):
        prompt = "Create a modern presentation about climate change for Instagram."
        slots = {"style": "modern", "content_type": "presentation", "topic": "climate change"}
        assert compute_slot_coverage(prompt, slots) == 1.0

    def test_partial_coverage(self):
        prompt = "Design a presentation about climate change."
        slots = {"style": "modern", "topic": "climate change", "platform": "Instagram"}
        score = compute_slot_coverage(prompt, slots)
        # Only "climate change" is present → 1/3
        assert 0.3 <= score <= 0.4

    def test_no_coverage(self):
        prompt = "Hello world."
        slots = {"topic": "quantum physics", "style": "elegant"}
        assert compute_slot_coverage(prompt, slots) == 0.0

    def test_empty_slots(self):
        prompt = "Any prompt text."
        slots = {"topic": None, "style": "", "platform": None}
        # No active slots → 1.0 (nothing to cover)
        assert compute_slot_coverage(prompt, slots) == 1.0

    def test_case_insensitive(self):
        prompt = "A MODERN design about CLIMATE CHANGE."
        slots = {"style": "modern", "topic": "climate change"}
        assert compute_slot_coverage(prompt, slots) == 1.0


# ---- Fluency Scoring ----

class TestFluencyScore:
    def test_empty_prompt(self):
        assert compute_fluency_score("") == 0.0
        assert compute_fluency_score("   ") == 0.0

    def test_optimal_length(self):
        # A well-formed prompt in the sweet-spot range
        prompt = (
            "Create a modern, visually engaging presentation about renewable energy. "
            "Include charts showing solar and wind growth trends. "
            "Use a professional color scheme with blue and green accents."
        )
        score = compute_fluency_score(prompt)
        assert 0.4 <= score <= 1.0

    def test_very_short_prompt(self):
        score = compute_fluency_score("Hi.")
        assert score < 0.6

    def test_repetitive_prompt_penalized(self):
        # Highly repetitive text
        prompt = "design design design design design design design design design design "
        score_repetitive = compute_fluency_score(prompt)

        good_prompt = "Create a sleek modern poster with vibrant colors and clean typography for the event."
        score_good = compute_fluency_score(good_prompt)

        assert score_repetitive < score_good

    def test_score_in_range(self):
        prompt = "Design an infographic about data science trends using pastel colors."
        score = compute_fluency_score(prompt)
        assert 0.0 <= score <= 1.0


# ---- Ensemble Score ----

class TestEnsembleScore:
    def test_default_weights(self):
        score = compute_ensemble_score(1.0, 1.0)
        assert abs(score - 1.0) < 1e-9

    def test_custom_weights(self):
        score = compute_ensemble_score(1.0, 0.0, alpha=0.7, beta=0.3)
        assert abs(score - 0.7) < 1e-9

    def test_zero_scores(self):
        assert compute_ensemble_score(0.0, 0.0) == 0.0

    def test_mixed_scores(self):
        score = compute_ensemble_score(0.8, 0.6, alpha=0.5, beta=0.5)
        assert abs(score - 0.7) < 1e-9


# ---- Select Best Prompt ----

class TestSelectBestPrompt:
    def test_picks_higher_scoring(self):
        candidates = [
            {"source": "offline", "text": "A brief note."},
            {
                "source": "gemini",
                "text": (
                    "Create a modern, visually engaging presentation about renewable energy. "
                    "Include charts showing growth trends in solar and wind power."
                ),
            },
        ]
        slots = {"topic": "renewable energy", "style": "modern"}
        best, log = select_best_prompt(candidates, slots)

        # The longer, more complete gemini version should score higher
        assert best["source"] == "gemini"
        assert len(log) == 2
        assert any(entry.get("selected") for entry in log)

    def test_scoring_log_structure(self):
        candidates = [{"source": "a", "text": "Test prompt text here."}]
        slots = {}
        best, log = select_best_prompt(candidates, slots)
        assert "slot_score" in log[0]
        assert "fluency_score" in log[0]
        assert "final_score" in log[0]
        assert "source" in log[0]

    def test_single_candidate(self):
        candidates = [{"source": "only", "text": "The only option."}]
        slots = {}
        best, log = select_best_prompt(candidates, slots)
        assert best["source"] == "only"


# ---- End-to-End ensemble_select ----

class TestEnsembleSelect:
    def test_returns_tuple_of_three(self):
        offline = "Create a modern presentation about AI."
        gemini = (
            "Design an engaging, visually stunning presentation about artificial intelligence. "
            "Include modern graphics and a clear narrative structure."
        )
        slots = {"topic": "AI", "style": "modern", "content_type": "presentation"}
        config = {"ensemble": {"alpha": 0.6, "beta": 0.4, "log_scores": False}}
        source, text, log = ensemble_select(offline, gemini, slots, config)

        assert source in ("offline", "gemini")
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(log, list)
        assert len(log) == 2

    def test_no_config_uses_defaults(self):
        source, text, log = ensemble_select(
            "Short offline.", "Short gemini.", {"topic": "test"}, None,
        )
        assert source in ("offline", "gemini")

    def test_slot_rich_prompt_preferred(self):
        """Prompt that covers more slots should be preferred."""
        offline = "Create a modern presentation about quantum computing for Instagram using pastel colors."
        gemini = "Here is a prompt."  # very short, no slot coverage
        slots = {
            "style": "modern",
            "topic": "quantum computing",
            "platform": "Instagram",
            "color_palette": "pastel",
        }
        config = {"ensemble": {"alpha": 0.8, "beta": 0.2, "log_scores": False}}
        source, text, log = ensemble_select(offline, gemini, slots, config)
        assert source == "offline"
