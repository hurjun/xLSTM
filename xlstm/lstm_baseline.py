"""A vanilla LSTM baseline with the same I/O contract as :class:`xLSTMModel`.

Used for an honest, apples-to-apples comparison on the synthetic demo task.
It deliberately uses ``torch.nn.LSTM`` (the classic sigmoid/tanh gated LSTM,
*no* exponential gating, *no* matrix memory) so the comparison isolates the
xLSTM contributions rather than implementation quirks.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["LSTMBaseline"]


class LSTMBaseline(nn.Module):
    """Embedding -> multi-layer LSTM -> linear head.

    Args:
        vocab_size: Number of token classes.
        embedding_dim: Embedding / hidden width.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm_out = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor, parallel: bool = True) -> Tensor:  # noqa: ARG002
        """Map token ids ``(batch, seq_len)`` to logits ``(B, L, vocab)``."""
        x = self.embedding(tokens)
        x, _ = self.lstm(x)
        x = self.norm_out(x)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
