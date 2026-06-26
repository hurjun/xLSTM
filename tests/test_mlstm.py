"""Shape, matrix-memory and parallel/recurrent equivalence tests for mLSTM."""

from __future__ import annotations

import pytest
import torch

from xlstm import mLSTMCell, mLSTMConfig


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_output_shape(num_heads: int) -> None:
    cfg = mLSTMConfig(input_size=12, hidden_size=24, num_heads=num_heads)
    cell = mLSTMCell(cfg)
    x = torch.randn(5, 9, 12)
    out, _ = cell.forward_parallel(x)
    assert out.shape == (5, 9, 24)


def test_matrix_memory_update_shape() -> None:
    """The matrix memory must be (num_heads, head_dim, head_dim) per batch."""
    cfg = mLSTMConfig(input_size=8, hidden_size=16, num_heads=4)  # head_dim = 4
    cell = mLSTMCell(cfg)
    _, (C, n, m) = cell.forward_recurrent(torch.randn(3, 6, 8))
    assert C.shape == (3, 4, 4, 4)  # (batch, heads, head_dim, head_dim)
    assert n.shape == (3, 4, 4)  # (batch, heads, head_dim)
    assert m.shape == (3, 4)  # (batch, heads)


def test_parallel_matches_recurrent() -> None:
    """The O(L^2) parallel form must equal the O(L) recurrence numerically."""
    torch.manual_seed(0)
    cfg = mLSTMConfig(input_size=16, hidden_size=32, num_heads=4)
    cell = mLSTMCell(cfg)
    x = torch.randn(4, 13, 16)
    out_par, _ = cell.forward_parallel(x)
    out_rec, _ = cell.forward_recurrent(x)
    assert torch.allclose(out_par, out_rec, atol=1e-5, rtol=1e-4)


def test_outputs_finite() -> None:
    cfg = mLSTMConfig(input_size=8, hidden_size=16, num_heads=2)
    cell = mLSTMCell(cfg)
    x = torch.randn(3, 7, 8)
    assert torch.isfinite(cell.forward_parallel(x)[0]).all()
    assert torch.isfinite(cell.forward_recurrent(x)[0]).all()
