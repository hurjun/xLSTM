"""sLSTM: scalar-memory LSTM with exponential gating and a stabilizer state.

This is a from-scratch re-implementation of the sLSTM cell from

    Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517

The sLSTM keeps a *scalar* memory cell per unit (like the classic LSTM) but
replaces the input gate (and optionally the forget gate) with an **exponential**
activation, and adds a **normalizer state** ``n_t`` plus a **stabilizer state**
``m_t``.  The stabilizer is a running max in log-space that makes the otherwise
unbounded ``exp`` gates numerically stable (a log-sum-exp style trick).

Recurrence (per scalar memory cell, with ``φ = tanh``)::

    z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)          # cell input
    ĩ_t = W_i x_t + R_i h_{t-1} + b_i                # input pre-activation
    f̃_t = W_f x_t + R_f h_{t-1} + b_f                # forget pre-activation
    o_t = sigmoid(W_o x_t + R_o h_{t-1} + b_o)       # output gate

    log f_t = logsigmoid(f̃_t)   (sigmoid forget)  or  f̃_t  (exponential forget)
    m_t = max(log f_t + m_{t-1}, ĩ_t)                # stabilizer (running max)
    i'_t = exp(ĩ_t - m_t)                            # stabilized input gate
    f'_t = exp(log f_t + m_{t-1} - m_t)              # stabilized forget gate

    c_t = f'_t * c_{t-1} + i'_t * z_t                # scalar cell state
    n_t = f'_t * n_{t-1} + i'_t                      # normalizer state
    h_t = o_t * c_t / max(|n_t|, exp(-m_t))          # stabilized readout

The recurrent weights ``R_*`` are **block-diagonal across heads** ("memory
mixing"): a unit only mixes with the other units inside its own head.  This is
the multi-head structure of the paper's sLSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import sLSTMConfig

__all__ = ["sLSTMCell"]


class sLSTMCell(nn.Module):
    """A multi-head sLSTM cell operating on a full sequence.

    Args:
        config: An :class:`~xlstm.config.sLSTMConfig`.

    Shapes:
        - input ``x``: ``(batch, seq_len, input_size)``
        - output ``h``: ``(batch, seq_len, hidden_size)``
    """

    def __init__(self, config: sLSTMConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Input projections for the four gates: (input_size -> hidden_size).
        # Stacked into one matrix for efficiency: order [z, i, f, o].
        self.weight_ih = nn.Linear(config.input_size, 4 * h, bias=True)

        # Block-diagonal recurrent (memory-mixing) weights, one (head_dim,
        # head_dim) block per head, for each of the four gates.
        self.weight_hh = nn.Parameter(
            torch.empty(4, self.num_heads, self.head_dim, self.head_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weights; bias the forget gate so it starts near 'keep'."""
        nn.init.xavier_uniform_(self.weight_ih.weight)
        nn.init.zeros_(self.weight_ih.bias)
        for k in range(4):
            for head in range(self.num_heads):
                nn.init.orthogonal_(self.weight_hh[k, head])
        # Forget gate (index 2) bias: a positive bias means "remember" at init.
        bias = self.weight_ih.bias.view(4, -1)
        with torch.no_grad():
            bias[2].fill_(1.0)

    def _recurrent(self, h_prev: Tensor) -> Tensor:
        """Apply the block-diagonal recurrent weights to ``h_{t-1}``.

        Args:
            h_prev: ``(batch, hidden_size)``.

        Returns:
            ``(batch, 4, hidden_size)`` recurrent pre-activations for [z,i,f,o].
        """
        b = h_prev.shape[0]
        h = h_prev.view(b, self.num_heads, self.head_dim)
        # einsum over the per-head (head_dim, head_dim) blocks for all 4 gates.
        out = torch.einsum("bhd,khed->bkhe", h, self.weight_hh)
        return out.reshape(b, 4, self.config.hidden_size)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Run the sLSTM over a sequence.

        Args:
            x: Input of shape ``(batch, seq_len, input_size)``.
            state: Optional initial ``(c, n, m, h)`` each ``(batch, hidden)``.

        Returns:
            ``(outputs, final_state)`` where ``outputs`` is
            ``(batch, seq_len, hidden_size)`` and ``final_state`` is the last
            ``(c, n, m, h)`` tuple.
        """
        b, t, _ = x.shape
        h_size = self.config.hidden_size
        device, dtype = x.device, x.dtype

        if state is None:
            c = torch.zeros(b, h_size, device=device, dtype=dtype)
            n = torch.zeros(b, h_size, device=device, dtype=dtype)
            m = torch.zeros(b, h_size, device=device, dtype=dtype)
            h = torch.zeros(b, h_size, device=device, dtype=dtype)
        else:
            c, n, m, h = state

        # Pre-compute all input contributions in one matmul: (b, t, 4*h).
        x_pre = self.weight_ih(x).view(b, t, 4, h_size)

        outputs = []
        for step in range(t):
            r_pre = self._recurrent(h)  # (b, 4, h)
            pre = x_pre[:, step] + r_pre  # (b, 4, h)
            z_pre, i_pre, f_pre, o_pre = pre.unbind(dim=1)

            z = torch.tanh(z_pre)
            o = torch.sigmoid(o_pre)

            # Forget gate: log(sigmoid) keeps the classic bounded gate, while
            # the raw pre-activation gives a fully exponential forget gate.
            log_f = (
                F.logsigmoid(f_pre)
                if self.config.forget_gate == "sigmoid"
                else f_pre
            )

            # Stabilizer: running max in log-space (prevents exp overflow).
            m_new = torch.maximum(log_f + m, i_pre)
            i_gate = torch.exp(i_pre - m_new)
            f_gate = torch.exp(log_f + m - m_new)

            c = f_gate * c + i_gate * z
            n = f_gate * n + i_gate
            # Stabilized normaliser: floor by exp(-m) (== 1 in the true domain).
            # A tiny eps prevents 0/0 when both |n| and exp(-m) underflow to 0
            # under extreme inputs (e.g. a huge exponential forget gate).
            denom = torch.maximum(n.abs(), torch.exp(-m_new)).clamp_min(1e-6)
            h = o * (c / denom)

            m = m_new
            outputs.append(h)

        out = torch.stack(outputs, dim=1)  # (b, t, h)
        out = self.dropout(out)
        return out, (c, n, m, h)
