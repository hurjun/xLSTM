"""Train a small xLSTM (and an LSTM baseline) on a long-range memory task.

Runs on CPU in a few minutes.  Produces real training curves and a final
accuracy table, and saves a comparison figure to ``assets/``.

Example::

    python scripts/train_demo.py --task recall --seq-len 64 --steps 2000

The script is deterministic given ``--seed`` so the README numbers are
reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow "python scripts/train_demo.py" without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xlstm import (  # noqa: E402
    LSTMBaseline,
    make_batch,
    task_num_classes,
    task_vocab_size,
    xLSTMConfig,
    xLSTMModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return (loss, accuracy) of ``model`` on a fixed eval batch."""
    model.eval()
    with torch.no_grad():
        logits = model(x)[:, -1, :]  # classify at the final position
        loss = nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return loss, acc


def train_model(
    model: nn.Module,
    *,
    name: str,
    task: str,
    seq_len: int,
    vocab_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
) -> dict:
    """Train one model and return its logged history."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    eval_gen = torch.Generator().manual_seed(99991)
    x_eval, y_eval = make_batch(task, 1024, seq_len, vocab_size, generator=eval_gen)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.CrossEntropyLoss()

    history = {"step": [], "train_loss": [], "eval_loss": [], "eval_acc": []}
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Training {name} ({n_params:,} params) ===")
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(task, batch_size, seq_len, vocab_size, generator=gen)
        logits = model(x)[:, -1, :]
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == 1:
            e_loss, e_acc = evaluate(model, x_eval, y_eval)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["eval_loss"].append(e_loss)
            history["eval_acc"].append(e_acc)
            print(
                f"  step {step:5d}/{steps}  train_loss={loss.item():.4f}  "
                f"eval_loss={e_loss:.4f}  eval_acc={e_acc:.3f}"
            )
    elapsed = time.time() - t0
    final_loss, final_acc = evaluate(model, x_eval, y_eval)
    print(f"  done in {elapsed:.1f}s  final eval_acc={final_acc:.3f}")
    return {
        "name": name,
        "params": n_params,
        "history": history,
        "final_acc": final_acc,
        "final_loss": final_loss,
        "seconds": elapsed,
    }


def plot_results(results: list[dict], task: str, seq_len: int, out_path: Path) -> None:
    """Save a loss + accuracy comparison figure (small PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
    colors = {"xLSTM": "#1f77b4", "LSTM": "#d62728"}
    for r in results:
        c = colors.get(r["name"])
        h = r["history"]
        ax_loss.plot(h["step"], h["eval_loss"], label=r["name"], color=c)
        ax_acc.plot(h["step"], h["eval_acc"], label=r["name"], color=c)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("eval loss (cross-entropy)")
    ax_loss.set_title("Eval loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    ax_acc.set_xlabel("training step")
    ax_acc.set_ylabel("eval accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_title("Eval accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()
    fig.suptitle(f"Task: {task}  (seq_len={seq_len})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="recall", choices=["recall", "parity", "majority"])
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baseline", action="store_true", help="Skip the LSTM baseline.")
    p.add_argument(
        "--figure", default=str(REPO_ROOT / "assets" / "training_curves.png")
    )
    p.add_argument("--metrics-json", default=str(REPO_ROOT / "assets" / "metrics.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    vocab = task_vocab_size(args.task, args.vocab_size)
    n_classes = task_num_classes(args.task, args.vocab_size)
    assert vocab == n_classes, "this demo assumes vocab == n_classes"

    results = []

    xlstm_cfg = xLSTMConfig(
        vocab_size=vocab,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        conv_kernel_size=4,
        dropout=0.0,
        tie_weights=False,
    )
    xlstm = xLSTMModel(xlstm_cfg)
    results.append(
        train_model(
            xlstm,
            name="xLSTM",
            task=args.task,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
        )
    )

    if not args.no_baseline:
        lstm = LSTMBaseline(
            vocab_size=vocab,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_blocks,
        )
        results.append(
            train_model(
                lstm,
                name="LSTM",
                task=args.task,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_every=args.eval_every,
                seed=args.seed,
            )
        )

    plot_results(results, args.task, args.seq_len, Path(args.figure))

    summary = {
        "task": args.task,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "steps": args.steps,
        "seed": args.seed,
        "models": [
            {
                "name": r["name"],
                "params": r["params"],
                "final_acc": r["final_acc"],
                "final_loss": r["final_loss"],
                "seconds": r["seconds"],
            }
            for r in results
        ],
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved metrics -> {args.metrics_json}")
    print("\n=== Final results ===")
    for r in results:
        print(f"  {r['name']:6s}  params={r['params']:>8,}  "
              f"final_acc={r['final_acc']:.3f}  final_loss={r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
