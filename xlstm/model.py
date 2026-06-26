"""Full stacked xLSTM model for sequence classification / language modelling."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .blocks import build_block
from .config import xLSTMConfig

__all__ = ["xLSTMModel"]


class xLSTMModel(nn.Module):
    """Token embedding -> stack of xLSTM blocks -> output head.

    The model returns per-position logits ``(batch, seq_len, vocab_size)``.  For
    sequence-classification tasks take the logits at the final position; for
    language modelling use every position.

    Args:
        config: An :class:`~xlstm.config.xLSTMConfig`.
    """

    def __init__(self, config: xLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(build_block(bc) for bc in config.blocks)
        self.norm_out = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, parallel: bool = True) -> Tensor:
        """Map token ids to logits.

        Args:
            tokens: Long tensor ``(batch, seq_len)`` of token ids.
            parallel: Use the parallel mLSTM form where available.

        Returns:
            Logits ``(batch, seq_len, vocab_size)``.
        """
        x = self.embed_dropout(self.embedding(tokens))
        for block in self.blocks:
            x = block(x, parallel=parallel)
        x = self.norm_out(x)
        return self.head(x)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> int:
    """Convenience: trainable parameter count of any module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
