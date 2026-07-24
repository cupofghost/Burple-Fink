"""Fine-tune the shared base model onto a single dataset (fine-tuning, stage 2).

Why this exists
---------------
Instead of learning to spell from scratch, we start from ``base.pt`` — which already
learned general name-spelling across every dataset in :mod:`src.pretrain` — and nudge
it toward one domain with a *low* learning rate for a *few* epochs. This reaches
good, on-style samples much faster than the Stage-0 from-scratch trainer, which is
the whole promise of transfer learning and the point of the platform.

The base and the fine-tune share one vocabulary (see HANDOFF §6), so the base's
embedding and output layers already have the right dimensions and its weights load
cleanly — no resizing, no vocab mismatch. We load the architecture/vocab straight
from the base checkpoint, override only the *training* hyperparameters (fewer epochs,
gentler learning rate), and save the result as ``checkpoints/<name>_ft.pt`` with
``training_names`` set to the fine-tune dataset so novelty is judged against *that*
domain.

Usage:
    python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models
    python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models --epochs 80 --lr 3e-4
"""

from __future__ import annotations

import argparse
import os

import torch

from .config import Config
from .data import filter_to_vocab, load_names
from .sample import load_checkpoint
from .train import fit, save_checkpoint

# Fine-tuning defaults: much shorter and gentler than pretraining, so the base
# model is *nudged* toward the domain rather than overwritten. Overridable on the CLI.
#
# Raised from the original 60 epochs / 5e-4 once the base spanned 13 wildly
# different domains (guns, dog breeds, bands, ...) instead of just two similar
# automotive ones: the gentler defaults left visible cross-domain leakage (e.g.
# dog-breed samples with beer/aircraft fragments in them). 150 epochs @ 2e-3
# specializes cleanly — verified via samples across every current dataset.
FINETUNE_EPOCHS = 150
FINETUNE_LR = 2e-3


def finetune(
    base_path: str,
    data_path: str,
    out_name: str,
    epochs: int | None = None,
    learning_rate: float | None = None,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
) -> str:
    # Load the base model together with the exact config + shared vocab it was built
    # with. Architecture fields must stay as-is or the loaded weights won't fit; we
    # only override the training knobs below.
    model, vocab, cfg, _base_names = load_checkpoint(base_path, device)
    cfg.epochs = epochs if epochs is not None else FINETUNE_EPOCHS
    cfg.learning_rate = learning_rate if learning_rate is not None else FINETUNE_LR

    names = load_names(data_path)
    names, dropped = filter_to_vocab(names, vocab)
    if not names:
        raise ValueError(
            f"None of the names in {data_path!r} are representable in the base model's "
            f"shared vocabulary. Add this dataset before pretraining (or rebuild the "
            f"shared vocab) so the base model knows its characters."
        )
    if dropped:
        print(f"  (skipped {len(dropped)} names with characters outside the base "
              f"model's shared vocab)")

    print(f"Fine-tuning {base_path} on {len(names)} names from {data_path} "
          f"| {cfg.epochs} epochs @ lr {cfg.learning_rate} | device {device}")

    fit(model, vocab, names, cfg, device=device, log_prefix="ft ")

    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}_ft.pt"), model, cfg, vocab, names
    )
    print(f"\nSaved fine-tuned checkpoint -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a shared base char-RNN on a single dataset.")
    parser.add_argument("--base", default="checkpoints/base.pt",
                        help="Base checkpoint to fine-tune from (see src.pretrain).")
    parser.add_argument("--data", required=True,
                        help="Dataset to specialize on (newline-separated names).")
    parser.add_argument("--name", required=True,
                        help="Output name; saved as checkpoints/<name>_ft.pt")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Fine-tune epochs (default {FINETUNE_EPOCHS}).")
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate",
                        help=f"Fine-tune learning rate (default {FINETUNE_LR}).")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    if not os.path.exists(args.base):
        parser.error(
            f"Base checkpoint not found: {args.base}. Run `python -m src.pretrain` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune(
        base_path=args.base,
        data_path=args.data,
        out_name=args.name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
