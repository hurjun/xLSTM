"""Self-contained synthetic sequence tasks that exercise long-range memory.

All generators return a batch of integer token sequences ``x`` of shape
``(batch, seq_len)`` and an integer label ``y`` of shape ``(batch,)`` that is
evaluated at the *final* time step.  Solving them requires carrying information
across the whole sequence, so they are good probes of a model's memory.

Tasks
-----
* ``recall`` -- "remember the first token": the label is the value of the very
  first token; every other token is a random distractor.  The model must
  transport one symbol across ``seq_len`` steps.
* ``parity`` -- the label is the parity (XOR / sum mod 2) of a binary sequence.
  Requires integrating information from *every* position.
* ``majority`` -- the label is the majority bit of a binary sequence.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor

__all__ = ["TASKS", "make_batch", "task_vocab_size"]

TASKS = ("recall", "parity", "majority")


def task_vocab_size(task: str, vocab_size: int) -> int:
    """Number of token classes the embedding needs for ``task``."""
    if task == "recall":
        return vocab_size
    if task in ("parity", "majority"):
        return 2
    raise ValueError(f"Unknown task {task!r}; choose from {TASKS}.")


def task_num_classes(task: str, vocab_size: int) -> int:
    """Number of output classes for ``task``."""
    if task == "recall":
        return vocab_size
    if task in ("parity", "majority"):
        return 2
    raise ValueError(f"Unknown task {task!r}; choose from {TASKS}.")


def make_batch(
    task: str,
    batch_size: int,
    seq_len: int,
    vocab_size: int = 10,
    generator: Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Generate one batch ``(x, y)`` for ``task``.

    Args:
        task: One of :data:`TASKS`.
        batch_size: Number of sequences.
        seq_len: Sequence length (the memory horizon).
        vocab_size: Token vocabulary for the ``recall`` task.
        generator: Optional :class:`torch.Generator` for reproducibility.
        device: Device for the returned tensors.

    Returns:
        ``(x, y)`` with ``x`` of shape ``(batch, seq_len)`` (long) and ``y`` of
        shape ``(batch,)`` (long).
    """
    g = generator
    if task == "recall":
        x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        y = x[:, 0].clone()
    elif task in ("parity", "majority"):
        x = torch.randint(0, 2, (batch_size, seq_len), generator=g)
        # parity: sum mod 2;  majority: 1 iff more than half the bits are 1.
        y = (
            x.sum(dim=1) % 2
            if task == "parity"
            else (x.sum(dim=1) * 2 > seq_len).long()
        )
    else:
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS}.")
    return x.to(device), y.to(device)
