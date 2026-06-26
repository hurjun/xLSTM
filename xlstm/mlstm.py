"""mLSTM: matrix-memory LSTM with exponential gating and a stabilizer state.

From-scratch re-implementation of the mLSTM cell from

    Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517

The mLSTM replaces the scalar memory of the classic LSTM with a **matrix memory**
``C_t`` updated by the *outer product* of a value and a key (a covariance /
fast-weight update).  Read-out is a query against that matrix memory -- the same
mechanics as linear attention.  Because the gates do not depend on ``h_{t-1}``,
the whole sequence can be processed **in parallel** (an attention-like form),
which is implemented here alongside the reference recurrence.

Per head (scalar gates ``i_t, f_t``; vectors ``q_t, k_t, v_t``)::

    ĩ_t = w_i x_t + b_i            f̃_t = w_f x_t + b_f
    log f_t = logsigmoid(f̃_t)
    m_t = max(log f_t + m_{t-1}, ĩ_t)             # stabilizer
    i'_t = exp(ĩ_t - m_t)         f'_t = exp(log f_t + m_{t-1} - m_t)

    C_t = f'_t * C_{t-1} + i'_t * (v_t k_t^T)     # matrix memory (outer product)
    n_t = f'_t * n_{t-1} + i'_t * k_t             # normalizer vector
    h_t = (C_t q_t) / max(|n_t^T q_t|, exp(-m_t)) # stabilized query read-out

The keys are scaled by ``1/sqrt(head_dim)`` as in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import mLSTMConfig

__all__ = ["mLSTMCell"]


class mLSTMCell(nn.Module):
    """A multi-head mLSTM cell with matrix memory.

    Args:
        config: An :class:`~xlstm.config.mLSTMConfig`.

    Shapes:
        - input ``x``: ``(batch, seq_len, input_size)``
        - output ``h``: ``(batch, seq_len, hidden_size)``
    """

    def __init__(self, config: mLSTMConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Query / key / value projections (key & query share head_dim here).
        self.q_proj = nn.Linear(config.input_size, h, bias=True)
        self.k_proj = nn.Linear(config.input_size, h, bias=True)
        self.v_proj = nn.Linear(config.input_size, h, bias=True)
        # Scalar input/forget gate pre-activations, one per head.
        self.i_gate = nn.Linear(config.input_size, self.num_heads, bias=True)
        self.f_gate = nn.Linear(config.input_size, self.num_heads, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise projections; bias forget gate towards 'remember'."""
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.i_gate.weight)
        nn.init.zeros_(self.f_gate.weight)
        # Input gate starts small/negative, forget gate starts positive (keep).
        nn.init.constant_(self.i_gate.bias, -2.0)
        nn.init.constant_(self.f_gate.bias, 1.0)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project inputs into per-head q, k, v and gate pre-activations."""
        b, t, _ = x.shape
        scale = self.head_dim**-0.5
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim) * scale
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim)
        i_pre = self.i_gate(x)  # (b, t, heads)
        f_pre = self.f_gate(x)  # (b, t, heads)
        return q, k, v, i_pre, f_pre

    def forward(
        self, x: Tensor, parallel: bool = True
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Run the mLSTM over a sequence.

        Args:
            x: Input of shape ``(batch, seq_len, input_size)``.
            parallel: If ``True`` use the O(L^2) parallel (attention-like) form;
                otherwise use the reference recurrence. Both are numerically
                equivalent up to floating point.

        Returns:
            ``(outputs, final_state)``. ``outputs`` is
            ``(batch, seq_len, hidden_size)``. ``final_state`` is the last
            ``(C, n, m)`` from the recurrence, or ``None`` placeholders when
            ``parallel=True`` (the parallel form does not materialise them).
        """
        if parallel:
            return self.forward_parallel(x)
        return self.forward_recurrent(x)

    def forward_recurrent(
        self, x: Tensor, state: tuple[Tensor, Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Reference O(L) recurrence with an explicit matrix memory."""
        b, t, _ = x.shape
        nh, hd = self.num_heads, self.head_dim
        device, dtype = x.device, x.dtype
        q, k, v, i_pre, f_pre = self._project(x)

        if state is None:
            C = torch.zeros(b, nh, hd, hd, device=device, dtype=dtype)
            n = torch.zeros(b, nh, hd, device=device, dtype=dtype)
            m = torch.zeros(b, nh, device=device, dtype=dtype)
        else:
            C, n, m = state

        log_f_all = F.logsigmoid(f_pre)  # (b, t, nh)
        outputs = []
        for step in range(t):
            log_f = log_f_all[:, step]  # (b, nh)
            i_p = i_pre[:, step]  # (b, nh)
            m_new = torch.maximum(log_f + m, i_p)  # (b, nh)
            i_gate = torch.exp(i_p - m_new)  # (b, nh)
            f_gate = torch.exp(log_f + m - m_new)  # (b, nh)

            kt = k[:, step]  # (b, nh, hd)
            vt = v[:, step]  # (b, nh, hd)
            qt = q[:, step]  # (b, nh, hd)
            outer = vt.unsqueeze(-1) * kt.unsqueeze(-2)  # (b, nh, hd, hd)

            C = f_gate[..., None, None] * C + i_gate[..., None, None] * outer
            n = f_gate[..., None] * n + i_gate[..., None] * kt

            num = torch.einsum("bhde,bhe->bhd", C, qt)  # C q
            denom = torch.einsum("bhd,bhd->bh", n, qt)  # n^T q
            denom = torch.maximum(denom.abs(), torch.exp(-m_new)).clamp_min(1e-6)
            h = num / denom[..., None]  # (b, nh, hd)

            m = m_new
            outputs.append(h.reshape(b, nh * hd))

        out = torch.stack(outputs, dim=1)  # (b, t, hidden)
        out = self.dropout(out)
        return out, (C, n, m)

    def forward_parallel(self, x: Tensor) -> tuple[Tensor, tuple]:
        """Parallel (attention-like) form, equivalent to the recurrence.

        Builds the lower-triangular gate-decay matrix ``D`` in log-space,
        stabilizes each row by its max, then forms the linear-attention output.
        Complexity is ``O(L^2 * head_dim)`` but fully parallel over time.
        """
        b, t, _ = x.shape
        nh, hd = self.num_heads, self.head_dim
        q, k, v, i_pre, f_pre = self._project(x)
        # Move heads to the batch dimension: (b, nh, t, hd).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        i_pre = i_pre.transpose(1, 2)  # (b, nh, t)
        log_f = F.logsigmoid(f_pre).transpose(1, 2)  # (b, nh, t)

        # Cumulative log-forget so that sum_{j<r<=s} log f_r = F_s - F_j.
        # F_s = sum_{r<=s} log f_r ; offset so F at the "0-th" boundary is 0.
        cum_f = torch.cumsum(log_f, dim=-1)  # (b, nh, t)

        # log D[s, j] = i_j + (F_s - F_j) for s >= j, else -inf.
        # F_s - F_j with the convention that f_j is *not* counted at its own j.
        log_D = (
            i_pre[:, :, None, :]  # i_j broadcast over s
            + cum_f[:, :, :, None]  # F_s
            - cum_f[:, :, None, :]  # F_j
        )  # (b, nh, t_s, t_j)
        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        log_D = log_D.masked_fill(~causal, float("-inf"))

        # Row-wise stabilizer.  To match the recurrence exactly we must also
        # include the "all-forget from the zero initial memory" path, whose
        # log-weight at time s is exactly the cumulative forget F_s = cum_f[s].
        row_max = log_D.max(dim=-1, keepdim=True).values  # (b, nh, t_s, 1)
        m = torch.maximum(row_max, cum_f[:, :, :, None])  # (b, nh, t_s, 1)
        D = torch.exp(log_D - m)  # (b, nh, t_s, t_j)

        # Linear-attention scores with the gate decay applied.
        scores = torch.einsum("bhsd,bhjd->bhsj", q, k) * D  # (b, nh, t_s, t_j)
        denom = scores.sum(dim=-1, keepdim=True).abs()  # |n^T q|
        denom = torch.maximum(denom, torch.exp(-m)).clamp_min(1e-6)
        out = torch.einsum("bhsj,bhjd->bhsd", scores, v) / denom  # (b, nh, t, hd)

        out = out.transpose(1, 2).reshape(b, t, nh * hd)
        out = self.dropout(out)
        return out, ()
