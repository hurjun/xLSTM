"""xLSTM: a from-scratch, educational PyTorch re-implementation.

Implements the building blocks of

    Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517

namely the scalar-memory ``sLSTM`` cell and the matrix-memory ``mLSTM`` cell
(both with exponential gating and a numerical stabilizer state), the pre-norm
residual blocks that wrap them, and a configurable stacked model.

This is an independent reimplementation for learning/portfolio purposes, not the
authors' official code, and not novel research.
"""

from __future__ import annotations

from .blocks import CausalConv1d, build_block, mLSTMBlock, sLSTMBlock
from .config import BlockConfig, mLSTMConfig, sLSTMConfig, xLSTMConfig
from .data import make_batch, task_num_classes, task_vocab_size
from .lstm_baseline import LSTMBaseline
from .mlstm import mLSTMCell
from .model import count_parameters, xLSTMModel
from .slstm import sLSTMCell

__version__ = "0.1.0"

__all__ = [
    "BlockConfig",
    "CausalConv1d",
    "LSTMBaseline",
    "build_block",
    "count_parameters",
    "make_batch",
    "mLSTMBlock",
    "mLSTMCell",
    "mLSTMConfig",
    "sLSTMBlock",
    "sLSTMCell",
    "sLSTMConfig",
    "task_num_classes",
    "task_vocab_size",
    "xLSTMConfig",
    "xLSTMModel",
]
