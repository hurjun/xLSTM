# xLSTM from scratch — PyTorch re-implementation of the xLSTM cells, tested on long-range recall

[![CI](https://github.com/hurjun/xLSTM/actions/workflows/ci.yml/badge.svg)](https://github.com/hurjun/xLSTM/actions/workflows/ci.yml) ![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

An independent, educational re-implementation of xLSTM (Beck et al., NeurIPS 2024, [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)). xLSTM is an LSTM with exponential gates (gates above 1) and a matrix memory (a d×d state per head). The repo builds both cells, their stabilizer, and the residual blocks. A small model trains on CPU in about 2 minutes. Not the official code, not new research.

![Recall accuracy vs sequence length: xLSTM vs LSTM](assets/recall_sweep.png)

## Result

All runs use a synthetic "remember the first token" task with vocabulary 16, so chance = 6.25%. Settings: AdamW lr 3e-3, cosine schedule, batch 64, seed 0, CPU.

- 64-token recall, 1500 steps: xLSTM **100.0%** (124,872 params); vanilla LSTM **6.8%** (68,736 params).
- Length sweep, 1100 steps: the LSTM solves 8 tokens, then sits at chance from 16 tokens on.
- Same sweep: the xLSTM holds **100%** at 8, 16, 32, 48 and 64 tokens.
- At 96 tokens the xLSTM reached 31.4% inside the 1100-step budget; a longer run is not committed.
- 35 CPU tests, CI on Python 3.10-3.12. The parallel mLSTM form matches the recurrence within `torch.allclose(atol=1e-5, rtol=1e-4)`.

**Why I built it.** Sequence models like this are a backbone of failure prediction from telemetry. Before trusting one in a reliability pipeline, I wanted to understand its numerics myself. Exponential gates overflow float32 unless a stabilizer state is used. So the tests feed inputs up to 1000x larger than normal and check that outputs stay finite. Gradients are checked at 100x.

## How it works

- **sLSTM cell**: one scalar memory per unit, exponential input gate, recurrence that mixes units only within a head.
- **mLSTM cell**: a matrix memory per head, updated by the outer product `v kᵀ` and read by a query. No hidden-to-hidden recurrence, so it also runs in a parallel linear-attention form.
- **Stabilizer**: `exp(89) = inf` in float32. A running-max state `m_t` and a normalizer `n_t` keep both gates in (0, 1]. Derivation in [NOTES.md](NOTES.md).
- **Blocks**: pre-norm residual blocks with causal conv, GroupNorm and gates; mLSTM and sLSTM blocks alternate.
- **Baseline**: a vanilla `nn.LSTM` with the same input/output contract, trained with the same settings.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_demo.py --task recall --seq-len 64 --vocab-size 16 --steps 1500   # headline run, ~2 min on CPU
python scripts/recall_sweep.py                                                        # length sweep, ~6.7 min on CPU
pytest -q                                                                             # 35 fast CPU tests
```

Requires Python 3.10 or newer. CPU only is fine. Each run overwrites `assets/*.png` and `assets/*.json`.

## Limitations

- Educational re-implementation of the paper's math, sized for a CPU demo (about 125k params, synthetic tasks). Not the official NX-AI code.
- The parallel mLSTM form is the plain O(L²) linear-attention version, not the paper's O(L) chunked kernel.
- Blocks keep the paper's structure but do not match every projection ratio and initialization.
- Not tested: whether a larger step budget lets the xLSTM reach 100% at 96 tokens.

## More

- [Details](docs/DETAILS.md): full result tables, the unrecorded larger-LSTM configs, architecture diagram, project tree, deviations from the paper, reproduce commands.
- [NOTES.md](NOTES.md): the exponential-gating and stabilizer derivation, sLSTM vs mLSTM vs LSTM.
- Paper: Beck et al., *xLSTM: Extended Long Short-Term Memory*, NeurIPS 2024 ([arXiv:2405.04517](https://arxiv.org/abs/2405.04517)). Official code (for reference, not used here): <https://github.com/NX-AI/xlstm>.

## License

MIT — this is an educational re-implementation; the xLSTM architecture is due to Beck et al.

— Jun Hur · [portfolio](https://hurjun.github.io) · [KCC 2024 paper](https://hurjun.github.io/asset/kcc2024-kiise-nursing-record-paper.pdf) · hurjun96@gmail.com
