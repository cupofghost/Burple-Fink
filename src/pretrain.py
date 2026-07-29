"""Pretrain one shared base model on *every* dataset at once (fine-tuning, stage 1).

Why this exists
---------------
Stage 0 trains a separate model per dataset from scratch, which throws away
everything the network learns about how names are spelled *in general*. Pretraining
a single base model on the union of all datasets teaches those shared regularities
once; :mod:`src.finetune` then specializes cheap copies of it per domain and reaches
good, on-style samples in far fewer epochs than training from scratch.

The base model is built against the **shared vocabulary** (see HANDOFF §6): every
checkpoint that will exchange weights — the base and each fine-tune — must agree on
the exact character<->id mapping, because the embedding and output layers are sized
to the vocabulary. The shared vocab is built (and persisted to
``data/shared_vocab.json``) the first time this runs, then reused.

Usage:
    python -m src.pretrain --epochs 300 --name base
    python -m src.pretrain --data-dir data --epochs 200 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import os

import torch

from .config import Config
from .data import (
    DEFAULT_VOCAB_PATH,
    build_shared_vocab,
    filter_to_vocab,
    list_dataset_files,
    load_all_names,
    load_shared_vocab,
    save_shared_vocab,
    split_names,
)
from .model import CharRNN
from .train import fit, save_checkpoint


def ensure_shared_vocab(paths, vocab_path: str = DEFAULT_VOCAB_PATH):
    """Load the persisted shared vocab, or build it from ``paths`` and save it.

    Building it once and reusing the file keeps the character<->id mapping stable
    across every future fine-tune, which is the whole point of a *shared* vocab.
    """
    if os.path.exists(vocab_path):
        vocab = load_shared_vocab(vocab_path)
        print(f"Loaded shared vocab ({len(vocab)} tokens) <- {vocab_path}")
    else:
        vocab = build_shared_vocab(paths)
        save_shared_vocab(vocab, vocab_path)
        print(f"Built shared vocab ({len(vocab)} tokens) -> {vocab_path}")
    return vocab


def pretrain(
    data_dir: str = "data",
    out_name: str = "base",
    cfg: Config | None = None,
    checkpoint_dir: str = "checkpoints",
    vocab_path: str = DEFAULT_VOCAB_PATH,
    device: str = "cpu",
) -> str:
    cfg = cfg or Config()

    paths = list_dataset_files(data_dir)
    if not paths:
        raise ValueError(f"No .txt datasets found in {data_dir!r}")

    vocab = ensure_shared_vocab(paths, vocab_path)

    names = load_all_names(paths)
    names, dropped = filter_to_vocab(names, vocab)
    if dropped:
        # Only possible if the vocab file predates a newly added dataset; rebuild it.
        print(f"  (skipped {len(dropped)} names with characters outside the shared "
              f"vocab — delete {vocab_path} to rebuild it)")
    train_names, val_names = split_names(names, cfg.val_fraction, cfg.seed)
    split_note = f" | {len(train_names)} train / {len(val_names)} val" if val_names else ""
    print(f"Pretraining on {len(names)} names from {len(paths)} datasets "
          f"| vocab {len(vocab)} | device {device}{split_note}")

    model = CharRNN(len(vocab), cfg, pad_id=vocab.pad_id).to(device)
    fit(model, vocab, train_names, cfg, device=device, val_names=val_names or None)

    out_path = save_checkpoint(
        os.path.join(checkpoint_dir, f"{out_name}.pt"), model, cfg, vocab,
        train_names, val_names,
    )
    print(f"\nSaved base checkpoint -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain a shared base char-RNN on all datasets at once.")
    parser.add_argument("--data-dir", default="data",
                        help="Directory of *.txt datasets to pretrain on.")
    parser.add_argument("--name", default="base",
                        help="Checkpoint name (without extension). Default: base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB_PATH,
                        help="Where to read/write the shared vocabulary JSON.")
    # --- WS-6 training-quality flags (default None => today's Config defaults) ---
    parser.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                        help="Hold out this share of the combined corpus for validation.")
    parser.add_argument("--patience", type=int, default=None, dest="early_stop_patience",
                        help="Stop after N epochs with no validation improvement.")
    parser.add_argument("--lr-schedule", choices=("none", "plateau", "cosine"),
                        default=None, dest="lr_schedule",
                        help="Learning-rate schedule. Default 'none' = constant LR.")
    args = parser.parse_args()

    cfg = Config()
    for field in ("epochs", "batch_size", "learning_rate", "hidden_dim",
                  "embedding_dim", "num_layers",
                  "val_fraction", "early_stop_patience", "lr_schedule"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain(
        data_dir=args.data_dir,
        out_name=args.name,
        cfg=cfg,
        checkpoint_dir=args.checkpoint_dir,
        vocab_path=args.vocab,
        device=device,
    )


if __name__ == "__main__":
    main()
