"""
tests/test_neural_ranker.py

Tests for Phase 2 neural ranker:
- PromptRankerNet output shape and range
- Forward pass with 384-d input
- Config loading
"""

import os
import sys

import numpy as np
import pytest
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_ranker import PromptRankerNet, load_config, _neural_cfg


class TestPromptRankerNet:
    def test_output_shape(self):
        """Output must be (batch_size, 1)."""
        model = PromptRankerNet(embed_dim=384, hidden_1=128, hidden_2=64, dropout=0.2)
        x = torch.randn(16, 384)
        out = model(x)
        assert out.shape == (16, 1)

    def test_output_range_sigmoid(self):
        """Sigmoid output must be in [0, 1]."""
        model = PromptRankerNet()
        x = torch.randn(32, 384)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_single_sample(self):
        """Single-sample forward pass works."""
        model = PromptRankerNet()
        x = torch.randn(1, 384)
        out = model(x)
        assert out.shape == (1, 1)

    def test_gradient_flows(self):
        """Backward pass computes gradients."""
        model = PromptRankerNet()
        x = torch.randn(4, 384)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check at least one parameter received a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad

    def test_custom_dims(self):
        """Non-default dimensions work."""
        model = PromptRankerNet(embed_dim=256, hidden_1=64, hidden_2=32, dropout=0.1)
        x = torch.randn(8, 256)
        out = model(x)
        assert out.shape == (8, 1)


class TestConfigLoading:
    def test_load_config_returns_dict(self):
        """config.yaml loads as dict with expected keys."""
        cfg = load_config("config.yaml")
        assert isinstance(cfg, dict)
        assert "seed" in cfg
        assert "neural_ranker" in cfg

    def test_neural_cfg_defaults(self):
        """Missing keys get safe defaults."""
        cfg = _neural_cfg({})
        assert cfg["embed_dim"] == 384
        assert cfg["hidden_1"] == 128
        assert cfg["hidden_2"] == 64
        assert cfg["dropout"] == 0.2
        assert cfg["lr"] == 1e-3
        assert cfg["batch_size"] == 32

    def test_neural_cfg_overrides(self):
        """User values override defaults."""
        cfg = _neural_cfg({"neural_ranker": {"lr": 0.01, "batch_size": 64}})
        assert cfg["lr"] == 0.01
        assert cfg["batch_size"] == 64
        assert cfg["hidden_1"] == 128  # default preserved
