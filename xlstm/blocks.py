"""xLSTM residual blocks: pre-norm wiring around the sLSTM / mLSTM cells.

Each block follows the pre-norm residual recipe ``x = x + f(LayerNorm(x))``.
The inner ``f`` up-projects the residual stream, optionally applies a causal
depthwise convolution (a short-range "smoothing" of recent tokens, as in the
paper's blocks), runs the recurrent cell, normalises per-head with GroupNorm,
applies an output gate, and down-projects back to the model width.

* mLSTM block -- "pre up-projection" style: project up, run matrix-memory
  attention-like cell, gate, project down.
* sLSTM block -- "post up-projection" style: run the scalar-memory cell at model
  width, GroupNorm, then a gated feed-forward up/down projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import BlockConfig, mLSTMConfig, sLSTMConfig
from .mlstm import mLSTMCell
from .slstm import sLSTMCell

__all__ = ["CausalConv1d", "mLSTMBlock", "sLSTMBlock", "build_block"]


class CausalConv1d(nn.Module):
    """Depthwise causal 1-D convolution over the time axis.

    Pads ``kernel_size - 1`` zeros on the left so position ``t`` only sees
    positions ``<= t`` (preserves autoregressive causality).

    Implemented as a manual shift-and-accumulate rather than a grouped
    :class:`torch.nn.Conv1d`, because the depthwise-conv backward is
    pathologically slow on CPU.  The two are mathematically identical.

    Shapes: ``(batch, seq_len, channels) -> (batch, seq_len, channels)``.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # weight[c, j] multiplies the input at relative offset (j - (k-1)).
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        bound = kernel_size**-0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        # Left-pad the time axis so output[t] sees inputs[t-(k-1) .. t].
        xp = F.pad(x, (0, 0, self.kernel_size - 1, 0))  # (b, t + k - 1, c)
        out = self.bias.expand(b, t, c).clone()
        for j in range(self.kernel_size):
            out = out + self.weight[:, j] * xp[:, j : j + t, :]
        return out


class mLSTMBlock(nn.Module):
    """Pre-norm residual block wrapping an :class:`mLSTMCell`."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config
        d = config.embedding_dim
        inner = int(config.proj_factor * d)
        self.norm = nn.LayerNorm(d)
        # Up-project to the main branch and a parallel gating branch.
        self.up_proj = nn.Linear(d, 2 * inner, bias=True)
        self.conv = (
            CausalConv1d(inner, config.conv_kernel_size)
            if config.conv_kernel_size > 0
            else None
        )
        self.cell = mLSTMCell(
            mLSTMConfig(
                input_size=inner,
                hidden_size=inner,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        )
        self.group_norm = nn.GroupNorm(config.num_heads, inner)
        self.down_proj = nn.Linear(inner, d, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, parallel: bool = True) -> Tensor:
        residual = x
        x = self.norm(x)
        up, gate = self.up_proj(x).chunk(2, dim=-1)  # each (b, t, inner)
        # The cell consumes the (optionally) convolved + activated main branch.
        cell_in = up
        if self.conv is not None:
            cell_in = F.silu(self.conv(up))
        h, _ = self.cell(cell_in, parallel=parallel)  # (b, t, inner)
        # Per-head normalisation (GroupNorm expects channels in dim 1).
        h = self.group_norm(h.transpose(1, 2)).transpose(1, 2)
        h = h * F.silu(gate)  # output gating
        out = self.down_proj(h)
        return residual + self.dropout(out)


class sLSTMBlock(nn.Module):
    """Pre-norm residual block wrapping an :class:`sLSTMCell`."""

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        self.config = config
        d = config.embedding_dim
        self.norm = nn.LayerNorm(d)
        self.conv = (
            CausalConv1d(d, config.conv_kernel_size)
            if config.conv_kernel_size > 0
            else None
        )
        self.cell = sLSTMCell(
            sLSTMConfig(
                input_size=d,
                hidden_size=d,
                num_heads=config.num_heads,
                forget_gate=config.forget_gate,
                dropout=config.dropout,
            )
        )
        self.group_norm = nn.GroupNorm(config.num_heads, d)
        # Gated feed-forward (up/down) projection after the cell.
        inner = int(config.proj_factor * d)
        self.ff_up = nn.Linear(d, 2 * inner, bias=True)
        self.ff_down = nn.Linear(inner, d, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, parallel: bool = True) -> Tensor:  # noqa: ARG002
        residual = x
        x = self.norm(x)
        cell_in = F.silu(self.conv(x)) if self.conv is not None else x
        h, _ = self.cell(cell_in)  # (b, t, d)
        h = self.group_norm(h.transpose(1, 2)).transpose(1, 2)
        # Gated GeLU feed-forward.
        a, b = self.ff_up(h).chunk(2, dim=-1)
        out = self.ff_down(F.gelu(a) * b)
        return residual + self.dropout(out)


def build_block(config: BlockConfig) -> nn.Module:
    """Factory: build the block module named by ``config.kind``."""
    if config.kind == "mlstm":
        return mLSTMBlock(config)
    if config.kind == "slstm":
        return sLSTMBlock(config)
    raise ValueError(f"Unknown block kind: {config.kind!r}")
