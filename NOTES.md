# Notes: sLSTM vs mLSTM vs vanilla LSTM, and the exponential-gating stabilizer

These notes explain the math behind this reimplementation of
[**xLSTM** (Beck et al., 2024)](https://arxiv.org/abs/2405.04517) and why the
*stabilizer state* is the crux that makes exponential gating usable.

---

## 1. The vanilla LSTM, in one line

The classic LSTM keeps a **scalar** memory cell `c_t` per unit and updates it
with **sigmoid** gates:

```
i_t = σ(ĩ_t)   f_t = σ(f̃_t)   o_t = σ(õ_t)   z_t = tanh(z̃_t)
c_t = f_t · c_{t-1} + i_t · z_t
h_t = o_t · tanh(c_t)
```

Two well-known limitations motivate xLSTM:

1. **No gate can ever exceed 1** (sigmoids saturate at 1). The cell cannot
   *sharpen* its decision to overwrite or strongly retain a memory; this limits
   how decisively it can "revise" stored information.
2. **One scalar cell per unit** has limited storage capacity, and the gates mix
   information sequentially (hard to parallelize).

xLSTM addresses (1) with **exponential gating** and (2) with a **matrix memory**.

---

## 2. Exponential gating and why it needs a stabilizer

xLSTM replaces the input gate (and optionally the forget gate) with an
**exponential** activation:

```
i_t = exp(ĩ_t)          # unbounded above!
f_t = σ(f̃_t)  or  exp(f̃_t)
```

`exp(ĩ_t)` can grow without bound, so `c_t` and the gates would overflow in
`float32` (`exp(89) = inf`). The fix is a **log-sum-exp style stabilizer**: an
extra state `m_t` that tracks the running maximum of the gates *in log-space*,
plus a **normalizer state** `n_t`. The key identity is that the read-out
`c_t / n_t` is **invariant** to a common rescaling of `c_t` and `n_t`, so we may
freely divide both by `exp(m_t)` to keep the numbers `O(1)`.

### sLSTM recurrence (scalar memory, implemented in `xlstm/slstm.py`)

For each scalar memory cell, with `φ = tanh`:

```
z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)        # cell input
ĩ_t =      W_i x_t + R_i h_{t-1} + b_i          # input pre-activation
f̃_t =      W_f x_t + R_f h_{t-1} + b_f          # forget pre-activation
o_t = σ(   W_o x_t + R_o h_{t-1} + b_o)         # output gate

log f_t = logσ(f̃_t)   (sigmoid forget)   or   f̃_t   (exponential forget)

m_t  = max(log f_t + m_{t-1},  ĩ_t)             # stabilizer (running max)
i'_t = exp(ĩ_t        - m_t)                     # stabilized input gate  ∈ (0, 1]
f'_t = exp(log f_t + m_{t-1} - m_t)             # stabilized forget gate ∈ (0, 1]

c_t = f'_t · c_{t-1} + i'_t · z_t               # scalar cell state
n_t = f'_t · n_{t-1} + i'_t                      # normalizer state
h_t = o_t · c_t / max(|n_t|, exp(-m_t))          # stabilized read-out
```

Because `m_t ≥ ĩ_t` and `m_t ≥ log f_t + m_{t-1}`, both `i'_t` and `f'_t` lie in
`(0, 1]` — they can **never overflow**. The denominator `max(|n_t|, exp(-m_t))`
floors the normalizer at the equivalent of `1` in the un-stabilized domain.

The recurrent matrices `R_*` are **block-diagonal across heads** ("memory
mixing"): a unit only mixes with units in its own head.

### mLSTM recurrence (matrix memory, implemented in `xlstm/mlstm.py`)

The mLSTM stores a **matrix** `C_t ∈ ℝ^{d×d}` updated by the **outer product** of
a value `v_t` and a key `k_t` (a fast-weight / covariance update), and reads it
out with a query `q_t` — exactly the mechanics of **linear attention**. The
gates `i_t, f_t` are scalar per head and depend only on `x_t` (not `h_{t-1}`),
which is what makes the whole sequence **parallelizable**:

```
ĩ_t = w_i x_t + b_i        f̃_t = w_f x_t + b_f       log f_t = logσ(f̃_t)
m_t  = max(log f_t + m_{t-1}, ĩ_t)
i'_t = exp(ĩ_t - m_t)      f'_t = exp(log f_t + m_{t-1} - m_t)

C_t = f'_t · C_{t-1} + i'_t · (v_t k_tᵀ)          # matrix memory (d×d)
n_t = f'_t · n_{t-1} + i'_t · k_t                  # normalizer vector (d)
h_t = (C_t q_t) / max(|n_tᵀ q_t|, exp(-m_t))       # stabilized query read-out
```

(keys are scaled by `1/√d`, as in the paper.)

### Parallel form (and why the implementation matches the recurrence exactly)

Because the gates do not depend on `h_{t-1}`, the unrolled contribution of step
`j` to the read-out at step `s ≥ j` has log-weight

```
log D[s, j] = ĩ_j + Σ_{r=j+1}^{s} log f_r = ĩ_j + (F_s − F_j),   F_s = Σ_{r≤s} log f_r
```

i.e. a lower-triangular **gate-decay matrix** `D`. Stabilizing each row `s` by

```
m_s = max( F_s ,  max_j log D[s, j] )
```

(the `F_s` term is the "all-forget from the zero initial memory" path, which is
exactly what the recurrence's `m_0 = 0` initialization contributes) gives a
single attention-like computation:

```
scores[s, j] = (q_sᵀ k_j) · exp(log D[s, j] − m_s)
h_s = Σ_j scores[s, j] · v_j  /  max(|Σ_j scores[s, j]|, exp(−m_s))
```

This O(L²) parallel form is **numerically identical** to the O(L) recurrence
(verified to `< 1e-6` in `tests/test_mlstm.py::test_parallel_matches_recurrent`).

---

## 3. sLSTM vs mLSTM vs LSTM — summary

| Property              | vanilla LSTM        | sLSTM                          | mLSTM                                   |
|-----------------------|---------------------|--------------------------------|-----------------------------------------|
| Memory                | scalar cell `c_t`   | scalar cell `c_t`              | **matrix** `C_t ∈ ℝ^{d×d}`              |
| Input/forget gating   | sigmoid             | **exp** input (+ stabilizer)   | **exp** input (+ stabilizer)            |
| Normalizer/stabilizer | none                | `n_t`, `m_t`                   | `n_t`, `m_t`                            |
| Recurrent `h_{t-1}`   | yes                 | yes (block-diagonal mixing)    | **no** (gates from `x_t` only)          |
| Parallel over time    | no                  | no                             | **yes** (linear-attention form)         |
| Memory update         | gated add           | gated add                      | outer-product `v_t k_tᵀ` (fast weights) |
| Read-out              | `o_t·tanh(c_t)`     | `o_t·c_t/n_t`                  | query `C_t q_t / (n_tᵀ q_t)`            |

In words: **sLSTM** keeps the LSTM's scalar memory but adds exponential gating +
stabilizer and a multi-head block-diagonal recurrence; it is good at *stateful,
sequential* computation. **mLSTM** swaps the scalar cell for an outer-product
matrix memory and drops the hidden-to-hidden recurrence, trading some
sequential expressivity for a much larger storage capacity and full
parallelism. xLSTM stacks both in pre-norm residual blocks.

---

## 4. Implementation deviations from the paper

This is an **educational** reimplementation; it is faithful to the core math but
deliberately simplified in a few places (see the README "Implementation notes"
section for the full list): a single shared head-dim for q/k/v, a manual
shift-and-accumulate causal conv (the grouped-conv backward is pathologically
slow on CPU), and small demo-scale models. None of these change the gating /
stabilizer mathematics above.
