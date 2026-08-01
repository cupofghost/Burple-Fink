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

The 60 epochs / 5e-4 below were always a guess. Since WS-6 you can measure instead:
``--val-fraction 0.15 --patience 10`` holds part of the fine-tune set back, keeps the
best-scoring epoch's weights, and stops when it stops improving. Off by default, so an
un-flagged fine-tune behaves exactly as it did before.

Usage:
    python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models
    python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models --epochs 80 --lr 3e-4
    python -m src.finetune --base checkpoints/base.pt --data data/car_models.txt --name car_models --val-fraction 0.15 --patience 10
"""

from __future__ import annotations

import argparse
import os

import torch

from .config import Config
from .data import filter_to_vocab, load_names, split_names
from .sample import load_checkpoint
from .train import add_regimen_args, apply_auto_epochs, fit, save_checkpoint

# Fine-tuning defaults: much shorter and gentler than pretraining, so the base
# model is *nudged* toward the domain rather than overwritten. Overridable on the CLI.
FINETUNE_EPOCHS = 60
FINETUNE_LR = 5e-4


def finetune(
    base_path: str,
    data_path: str,
    out_name: str,
    epochs: int | None = None,
    learning_rate: float | None = None,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    val_fraction: float | None = None,
    early_stop_patience: int | None = None,
    lr_schedule: str | None = None,
    weight_decay: float | None = None,
    label_smoothing: float | None = None,
    warmup_epochs: int | None = None,
    arch: str | None = None,
    auto_epochs: bool = False,
) -> str:
    """Fine-tune ``base_path`` onto ``data_path`` and write ``<name>_ft.pt``.

    ``arch`` is accepted for symmetry with the other two entry points, but it can only
    *confirm* the base checkpoint's architecture — the weights being loaded are that
    architecture's weights, so asking to fine-tune an LSTM base as a transformer is a
    mistake, not a request, and it is rejected rather than silently mislabelled.
    """
    # Load the base model together with the exact config + shared vocab it was built
    # with. Architecture fields must stay as-is or the loaded weights won't fit; we
    # only override the training knobs below.
    model, vocab, cfg, _base_names = load_checkpoint(base_path, device)
    if arch is not None and arch != cfg.arch:
        raise ValueError(
            f"--arch {arch} does not match the base checkpoint's architecture "
            f"({cfg.arch!r}). Fine-tuning starts from the base model's weights, so the "
            f"architecture is fixed by {base_path}. Pretrain a {arch} base first."
        )
    cfg.epochs = epochs if epochs is not None else FINETUNE_EPOCHS
    cfg.learning_rate = learning_rate if learning_rate is not None else FINETUNE_LR

    # WS-6 knobs are reset to the Config defaults (all off) unless explicitly passed,
    # rather than inherited from the base checkpoint's config: how much of the *base
    # corpus* was held out says nothing about this dataset, and silently splitting a
    # 66-name fine-tune set because the base run used 15% would be a nasty surprise.
    # This is also what keeps fine-tuning byte-identical to its pre-WS-6 behavior.
    defaults = Config()
    cfg.val_fraction = val_fraction if val_fraction is not None else defaults.val_fraction
    cfg.early_stop_patience = (early_stop_patience if early_stop_patience is not None
                               else defaults.early_stop_patience)
    cfg.lr_schedule = lr_schedule if lr_schedule is not None else defaults.lr_schedule

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

    train_names, val_names = split_names(names, cfg.val_fraction, cfg.seed)
    split_note = f" | {len(train_names)} train / {len(val_names)} val" if val_names else ""
    print(f"Fine-tuning {base_path} on {len(names)} names from {data_path} "
          f"| {cfg.epochs} epochs @ lr {cfg.learning_rate} | device {device}{split_note}")

    fit(model, vocab, train_names, cfg, device=device, log_prefix="ft ",
        val_names=val_names or None)

    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}_ft.pt"), model, cfg, vocab,
        train_names, val_names,
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
    # --- WS-6 training-quality flags. Fine-tuning is where these pay off most: the
    # 60 epochs / 5e-4 above are a guess, and a held-out slice plus --patience turns
    # that guess into a measurement. All default to off, as before. ---
    parser.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                        help="Hold out this share of the fine-tune set for validation "
                             "(e.g. 0.15). Default 0 = train on everything, as before.")
    parser.add_argument("--patience", type=int, default=None, dest="early_stop_patience",
                        help="Stop after N epochs with no validation improvement "
                             "(needs --val-fraction). Default 0 = never stop early.")
    parser.add_argument("--lr-schedule", choices=("none", "plateau", "cosine"),
                        default=None, dest="lr_schedule",
                        help="Learning-rate schedule. Default 'none' = constant LR.")
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
        val_fraction=args.val_fraction,
        early_stop_patience=args.early_stop_patience,
        lr_schedule=args.lr_schedule,
    )


if __name__ == "__main__":
    main()
