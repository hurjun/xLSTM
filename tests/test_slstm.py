"""Shape and behaviour tests for the sLSTM cell."""

from __future__ import annotations

import pytest
import torch

from xlstm import sLSTMCell, sLSTMConfig


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_output_shape(num_heads: int) -> None:
    cfg = sLSTMConfig(input_size=12, hidden_size=24, num_heads=num_heads)
    cell = sLSTMCell(cfg)
    x = torch.randn(5, 9, 12)
    out, state = cell(x)
    assert out.shape == (5, 9, 24)
    c, n, m, h = state
    for s in (c, n, m, h):
        assert s.shape == (5, 24)


def test_output_is_finite() -> None:
    cfg = sLSTMConfig(input_size=8, hidden_size=16, num_heads=4)
    cell = sLSTMCell(cfg)
    out, _ = cell(torch.randn(3, 7, 8))
    assert torch.isfinite(out).all()


def test_block_diagonal_recurrence_is_head_local() -> None:
    """A unit's recurrent input must only depend on its own head."""
    cfg = sLSTMConfig(input_size=4, hidden_size=8, num_heads=2)  # head_dim = 4
    cell = sLSTMCell(cfg)
    h = torch.randn(1, 8, requires_grad=True)
    rec = cell._recurrent(h)  # (1, 4, 8)
    # Gate value of head-0 unit 0 must not depend on head-1 inputs (units 4..7).
    grad = torch.autograd.grad(rec[0, 0, 0], h, retain_graph=True)[0]
    assert grad[0, 4:].abs().max() == 0.0


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        sLSTMConfig(input_size=8, hidden_size=10, num_heads=4)  # not divisible
    with pytest.raises(ValueError):
        sLSTMConfig(input_size=8, hidden_size=8, forget_gate="relu")
