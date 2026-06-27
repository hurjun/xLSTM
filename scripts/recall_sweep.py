"""Recall-length scaling sweep: accuracy vs sequence length, xLSTM vs LSTM.

Trains a small xLSTM and a vanilla ``nn.LSTM`` baseline on the ``recall`` task
("remember the first token") across a range of sequence lengths and records the
**real** final eval accuracy of each.  The point is to find the horizon at which
the classic LSTM collapses to chance while xLSTM still holds.

Everything is CPU-only, seeded, and sized so the whole sweep finishes in a few
minutes.  It reuses the exact training loop from ``train_demo.py`` (same
optimizer, schedule, batch size and seed), so the per-length runs are directly
comparable to the headline demo.

Outputs (written to ``assets/``):
* ``recall_sweep.png``  -- accuracy-vs-length line plot (xLSTM vs LSTM).
* ``recall_sweep.json`` -- the raw numbers behind the plot/table.

Reproduce with::

    python scripts/recall_sweep.py

Smaller/faster or longer/slower sweeps::

    python scripts/recall_sweep.py --seq-lens 8 16 32 64 --steps 800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Allow "python scripts/recall_sweep.py" without installing the package, and
# let us reuse the demo's training loop verbatim (same recipe, no duplication).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_demo import train_model  # noqa: E402

from xlstm import (  # noqa: E402
    LSTMBaseline,
    task_num_classes,
    task_vocab_size,
    xLSTMConfig,
    xLSTMModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_xlstm(vocab: int, embedding_dim: int, num_blocks: int, num_heads: int) -> xLSTMModel:
    """Fresh xLSTM with the same shape as the headline demo."""
    cfg = xLSTMConfig(
        vocab_size=vocab,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        conv_kernel_size=4,
        dropout=0.0,
        tie_weights=False,
    )
    return xLSTMModel(cfg)


def build_lstm(vocab: int, embedding_dim: int, num_blocks: int) -> LSTMBaseline:
    """Fresh LSTM baseline with the same I/O contract."""
    return LSTMBaseline(vocab_size=vocab, embedding_dim=embedding_dim, num_layers=num_blocks)


def plot_sweep(rows: list[dict], chance: float, out_path: Path) -> None:
    """Save the accuracy-vs-length comparison figure (small PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = [r["seq_len"] for r in rows]
    xlstm_acc = [r["xlstm_acc"] for r in rows]
    lstm_acc = [r["lstm_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=120)
    ax.plot(lengths, xlstm_acc, "o-", color="#1f77b4", lw=2, ms=6, label="xLSTM")
    ax.plot(lengths, lstm_acc, "s--", color="#d62728", lw=2, ms=6, label="LSTM")
    ax.axhline(chance, color="gray", ls=":", lw=1.2, label=f"chance (1/{round(1 / chance)})")
    ax.set_xlabel("sequence length (memory horizon)")
    ax.set_ylabel("final eval accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(lengths)
    ax.set_title("Recall task: accuracy vs sequence length")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[8, 16, 32, 48, 64, 96])
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=1100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figure", default=str(REPO_ROOT / "assets" / "recall_sweep.png"))
    p.add_argument("--metrics-json", default=str(REPO_ROOT / "assets" / "recall_sweep.json"))
    args = p.parse_args()

    task = "recall"
    vocab = task_vocab_size(task, args.vocab_size)
    n_classes = task_num_classes(task, args.vocab_size)
    assert vocab == n_classes, "this sweep assumes vocab == n_classes"
    chance = 1.0 / args.vocab_size

    t_start = time.time()
    rows: list[dict] = []
    for seq_len in args.seq_lens:
        print(f"\n########## sequence length {seq_len} ##########")
        # Seed the global RNG before *building* each model so weight init is
        # identical and deterministic at every length (train_model re-seeds the
        # data stream); this makes the whole sweep reproducible run-to-run.
        torch.manual_seed(args.seed)
        xlstm = build_xlstm(vocab, args.embedding_dim, args.num_blocks, args.num_heads)
        xr = train_model(
            xlstm,
            name="xLSTM",
            task=task,
            seq_len=seq_len,
            vocab_size=args.vocab_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        torch.manual_seed(args.seed)
        lstm = build_lstm(vocab, args.embedding_dim, args.num_blocks)
        lr = train_model(
            lstm,
            name="LSTM",
            task=task,
            seq_len=seq_len,
            vocab_size=args.vocab_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        rows.append(
            {
                "seq_len": seq_len,
                "xlstm_acc": xr["final_acc"],
                "xlstm_loss": xr["final_loss"],
                "xlstm_params": xr["params"],
                "lstm_acc": lr["final_acc"],
                "lstm_loss": lr["final_loss"],
                "lstm_params": lr["params"],
            }
        )

    plot_sweep(rows, chance, Path(args.figure))

    summary = {
        "task": task,
        "vocab_size": args.vocab_size,
        "chance": chance,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "embedding_dim": args.embedding_dim,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "total_seconds": round(time.time() - t_start, 1),
        "rows": rows,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved metrics -> {args.metrics_json}")

    print("\n=== Recall-length sweep (final eval accuracy) ===")
    print(f"{'seq_len':>8} | {'xLSTM':>7} | {'LSTM':>7}")
    print("-" * 30)
    for r in rows:
        print(f"{r['seq_len']:>8} | {r['xlstm_acc']:>7.3f} | {r['lstm_acc']:>7.3f}")
    print(f"\nTotal wall time: {summary['total_seconds']:.0f}s on CPU.")


if __name__ == "__main__":
    main()
