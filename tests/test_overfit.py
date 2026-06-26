"""End-to-end learning test: the blocks can overfit a single batch.

This proves the forward/backward wiring actually lets gradients drive the loss
down (a strong smoke test that the implementation *learns*, not just runs).
"""

from __future__ import annotations

import pytest
import torch

from xlstm import LSTMBaseline, make_batch, xLSTMConfig, xLSTMModel


@pytest.mark.parametrize("block_pattern", [["mlstm"], ["slstm"], ["mlstm", "slstm"]])
def test_overfit_single_batch(block_pattern: list[str]) -> None:
    torch.manual_seed(0)
    vocab, seq_len = 8, 16
    cfg = xLSTMConfig(
        vocab_size=vocab,
        embedding_dim=32,
        num_blocks=len(block_pattern),
        block_pattern=block_pattern,
        num_heads=4,
        tie_weights=False,
    )
    model = xLSTMModel(cfg)
    gen = torch.Generator().manual_seed(0)
    x, y = make_batch("recall", batch_size=16, seq_len=seq_len, vocab_size=vocab, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    for _ in range(120):
        logits = model(x)[:, -1, :]
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.1 * losses[0], (
        f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    acc = (model(x)[:, -1, :].argmax(-1) == y).float().mean().item()
    assert acc > 0.95


def test_finite_grads_after_backward() -> None:
    torch.manual_seed(0)
    cfg = xLSTMConfig(vocab_size=8, embedding_dim=32, num_blocks=2, tie_weights=False)
    model = xLSTMModel(cfg)
    tokens = torch.randint(0, 8, (4, 12))
    logits = model(tokens)
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_lstm_baseline_overfits() -> None:
    """The baseline shares the same training contract and can also overfit."""
    torch.manual_seed(0)
    model = LSTMBaseline(vocab_size=8, embedding_dim=32, num_layers=1)
    gen = torch.Generator().manual_seed(0)
    x, y = make_batch("recall", 16, 16, 8, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    first = last = None
    for i in range(150):
        loss = loss_fn(model(x)[:, -1, :], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
