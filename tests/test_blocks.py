"""Tests for the residual blocks and the full stacked model."""

from __future__ import annotations

import pytest
import torch

from xlstm import (
    BlockConfig,
    CausalConv1d,
    build_block,
    xLSTMConfig,
    xLSTMModel,
)


@pytest.mark.parametrize("kind", ["slstm", "mlstm"])
def test_block_preserves_shape(kind: str) -> None:
    cfg = BlockConfig(kind=kind, embedding_dim=32, num_heads=4)
    block = build_block(cfg)
    x = torch.randn(3, 10, 32)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_causal_conv_is_causal() -> None:
    """output[t] must not depend on inputs at positions > t."""
    conv = CausalConv1d(channels=6, kernel_size=4)
    x = torch.randn(2, 8, 6, requires_grad=True)
    out = conv(x)
    grad = torch.autograd.grad(out[:, 3].sum(), x, retain_graph=True)[0]
    assert grad[:, 4:].abs().max() == 0.0  # no leakage from the future


def test_model_forward_shape() -> None:
    cfg = xLSTMConfig(vocab_size=20, embedding_dim=32, num_blocks=4, num_heads=4)
    model = xLSTMModel(cfg)
    tokens = torch.randint(0, 20, (3, 11))
    logits = model(tokens)
    assert logits.shape == (3, 11, 20)
    assert torch.isfinite(logits).all()


def test_default_block_pattern_alternates() -> None:
    cfg = xLSTMConfig(vocab_size=10, num_blocks=4)
    assert cfg.block_pattern == ["mlstm", "slstm", "mlstm", "slstm"]


def test_custom_block_pattern() -> None:
    cfg = xLSTMConfig(vocab_size=10, num_blocks=3, block_pattern=["slstm"] * 3)
    model = xLSTMModel(cfg)
    tokens = torch.randint(0, 10, (2, 5))
    assert model(tokens).shape == (2, 5, 10)
    with pytest.raises(ValueError):
        xLSTMConfig(vocab_size=10, num_blocks=3, block_pattern=["slstm"])
