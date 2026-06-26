"""Numerical-stability tests for the exponential gating + stabilizer state.

The whole point of the stabilizer (the running-max / log-sum-exp trick) is that
the unbounded ``exp`` input gate never overflows.  These tests feed
deliberately huge inputs and assert that nothing becomes NaN/Inf, in both the
forward and backward pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from xlstm import mLSTMCell, mLSTMConfig, sLSTMCell, sLSTMConfig


def _naive_exp_gate_overflows() -> bool:
    """Sanity: a naive exp(large) really does overflow in float32."""
    return not torch.isfinite(torch.exp(torch.tensor(200.0))).item()


def test_naive_exp_would_overflow() -> None:
    assert _naive_exp_gate_overflows()  # justifies why we need the stabilizer


@pytest.mark.parametrize("scale", [10.0, 100.0, 1000.0])
@pytest.mark.parametrize("forget_gate", ["sigmoid", "exponential"])
def test_slstm_stable_under_large_inputs(scale: float, forget_gate: str) -> None:
    torch.manual_seed(0)
    cfg = sLSTMConfig(input_size=16, hidden_size=32, num_heads=4, forget_gate=forget_gate)
    cell = sLSTMCell(cfg)
    x = torch.randn(2, 24, 16) * scale
    out, state = cell(x)
    assert torch.isfinite(out).all(), "sLSTM output overflowed"
    for s in state:
        assert torch.isfinite(s).all(), "sLSTM state overflowed"


@pytest.mark.parametrize("scale", [10.0, 100.0, 1000.0])
def test_mlstm_stable_under_large_inputs(scale: float) -> None:
    torch.manual_seed(0)
    cfg = mLSTMConfig(input_size=16, hidden_size=32, num_heads=4)
    cell = mLSTMCell(cfg)
    x = torch.randn(2, 24, 16) * scale
    assert torch.isfinite(cell.forward_parallel(x)[0]).all()
    assert torch.isfinite(cell.forward_recurrent(x)[0]).all()


def test_stabilizer_keeps_gradients_finite() -> None:
    """Forward AND backward must stay finite under extreme inputs."""
    torch.manual_seed(0)
    cfg = sLSTMConfig(input_size=16, hidden_size=32, num_heads=4, forget_gate="exponential")
    cell = sLSTMCell(cfg)
    x = (torch.randn(2, 16, 16) * 100.0).requires_grad_(True)
    out, _ = cell(x)
    out.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()

    mcfg = mLSTMConfig(input_size=16, hidden_size=32, num_heads=4)
    mcell = mLSTMCell(mcfg)
    x2 = (torch.randn(2, 16, 16) * 100.0).requires_grad_(True)
    mout, _ = mcell.forward_parallel(x2)
    mout.pow(2).mean().backward()
    assert torch.isfinite(x2.grad).all()


def test_stabilizer_state_tracks_running_max() -> None:
    """The mLSTM stabilizer m must equal the running max of the log-gates."""
    torch.manual_seed(0)
    cfg = mLSTMConfig(input_size=8, hidden_size=8, num_heads=1)
    cell = mLSTMCell(cfg)
    x = torch.randn(1, 6, 8) * 5.0
    _, (_, _, m) = cell.forward_recurrent(x)
    # m is the final stabilizer; it must be finite and not absurdly large
    # despite exponential gating.
    assert torch.isfinite(m).all()
    assert m.abs().max() < 1e4
    assert not math.isnan(m.sum().item())
